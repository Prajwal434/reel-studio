"""Reel Studio engine. The app runs:  engine.py <command> <project_dir> [options]
Every stdout line is one JSON event: {"type": "progress"|"log"|"done"|"error", ...}.

Commands
  check                      tool/model availability (no project)
  llm-test                   one tiny request with the configured provider
  analyze <dir>              probe, extract audio, (cover) separate voice/instrument, transcribe, build lines + trim
  still <dir> <look> <t>     graded preview frame -> <dir>/stills/<look>.jpg
  render <dir> ig|yt         final video -> <dir>/exports/
  caption <dir> <platform>   post caption (LLM if configured, template otherwise)
  suggest-replies <dir>      (skit) AI drafts what the AI character says in each pause
  voice <dir> <i>            (skit) preview reply i in the AI voice
  setup                      download all models (first run)

project.json: name, video, template (cover|talk|skit|montage|custom; the last two take clips[] instead of video), song{title,artist}, look, duration, fps, trim{in,out,fade_out},
lines[{start,end,text}], title_card, inst_db."""
import sys, os, json, subprocess, shutil, re, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if os.environ.get("RS_BIN"):  # bundled app: its own ffmpeg first
    os.environ["PATH"] = os.environ["RS_BIN"] + os.pathsep + os.environ.get("PATH", "")
SEPARATOR = [sys.executable, "-c", "import sys; from audio_separator.utils.cli import main; sys.exit(main())"]
W, H, FPS = 1080, 1920, 30
MODELS = os.environ.get("RS_MODELS") or os.path.join(HERE, "..", "..", "tools", "models")
WHISPER_DIR = os.environ.get("RS_WHISPER_DIR") or None  # None = Hugging Face cache
SEP_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def emit(type_, **kw):
    print(json.dumps({"type": type_, **kw}), flush=True)


def progress(step, pct, msg=""):
    emit("progress", step=step, pct=round(float(pct), 3), msg=msg)


def load(d):
    with open(os.path.join(d, "project.json")) as f: return json.load(f)


def save(d, p):
    tmp = os.path.join(d, "project.json.tmp")
    with open(tmp, "w") as f: json.dump(p, f, indent=2)
    os.replace(tmp, os.path.join(d, "project.json"))


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: raise RuntimeError(f"{os.path.basename(cmd[0])} failed: {r.stderr.strip()[-400:]}")
    return r.stdout


def probe(path):
    """duration, display size (rotation applied), fps, audio? via PyAV (no ffprobe needed)"""
    import av
    with av.open(path) as c:
        v = next((st for st in c.streams if st.type == "video"), None)
        if v is None: raise RuntimeError("No video stream in this file.")
        w, h = v.codec_context.width, v.codec_context.height
        rot = 0
        try:
            for fr in c.decode(video=0):
                rot = abs(int(round(getattr(fr, "rotation", 0) or 0))); break
        except Exception: pass
        if rot % 180 == 90: w, h = h, w
        dur = float(c.duration / 1e6) if c.duration else float(v.duration * v.time_base)
        fps = float(v.average_rate) if v.average_rate else 30.0
        return dict(duration=dur, width=w, height=h, fps=round(fps, 3), has_audio=any(st.type == "audio" for st in c.streams))


# ---------------------------------------------------------------- analyze
def transcribe(path, prompt=None):
    from faster_whisper import WhisperModel
    import soundfile as sf, librosa
    a, sr = sf.read(path, always_2d=True)
    y = librosa.resample(a.mean(1).astype(np.float32), orig_sr=sr, target_sr=16000)
    m = WhisperModel("small", device="cpu", compute_type="int8", download_root=WHISPER_DIR)
    segs, _ = m.transcribe(y, word_timestamps=True, vad_filter=False, condition_on_previous_text=False, temperature=0, initial_prompt=prompt)
    words = []
    for s in segs:
        for w in s.words or []:
            words.append(dict(start=float(w.start), end=float(w.end), word=w.word.strip()))
        progress("transcribe", min(0.99, s.end / (len(y) / 16000)))
    # drop hallucinated runs ("ah ah ah ...", "ooooo")
    out, run_ = [], 0
    for i, w in enumerate(words):
        run_ = run_ + 1 if i and w["word"].lower() == words[i - 1]["word"].lower() else 0
        if run_ >= 3 or re.fullmatch(r"(o{3,}|a{3,}|m{3,})\W*", w["word"].lower()): continue
        out.append(w)
    return out


FILLER = re.compile(r"^(ah+|oh+|ooh+|mm+|hm+|la+|na+|uh+|ha+)\W*$", re.I)


def phrases(vocal_wav):
    """sung phrases from vocal energy (more reliable than whisper's timings on held notes)"""
    import soundfile as sf
    v, sr = sf.read(vocal_wav, always_2d=True); v = v.mean(1)
    h = int(0.1 * sr)
    db = np.array([20 * np.log10(np.sqrt(np.mean(v[i:i + h] ** 2)) + 1e-9) for i in range(0, len(v) - h, h)])
    on = db > db.max() - 30
    segs, s = [], None
    for i, o in enumerate(list(on) + [False]):
        if o and s is None: s = i
        if not o and s is not None:
            if i - s >= 2: segs.append([s / 10, i / 10])
            s = None
    merged = []
    for a, b in segs:
        if merged and a - merged[-1][1] < 0.3: merged[-1][1] = b
        else: merged.append([a, b])
    return merged


def cover_lines(words, segs):
    lines = [dict(start=a, end=b, words=[]) for a, b in segs]
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        best = min(lines, key=lambda l: 0 if l["start"] - 0.3 <= mid <= l["end"] + 0.3 else min(abs(mid - l["start"]), abs(mid - l["end"])), default=None)
        if best and (best["start"] - 1.0 <= mid <= best["end"] + 1.0): best["words"].append(w["word"])
    out = []
    for l in lines:
        text = " ".join(x for x in l["words"] if not FILLER.match(x)).strip()
        if not text: continue  # wordless "ahh" / humming: no text on screen
        text = text[0].upper() + text[1:]
        text = text.rstrip(".,")
        ws = text.split()  # whisper often doubles a phrase across a held note: "X? X?" -> "X?"
        for k in range(len(ws) // 2, 1, -1):
            if len(ws) == 2 * k and [w.lower().strip("?,.!") for w in ws[:k]] == [w.lower().strip("?,.!") for w in ws[k:]]:
                text = " ".join(ws[:k]); break
        if out and l["start"] - out[-1]["end"] <= 0.55 and len(out[-1]["text"]) + len(text) <= 40:
            out[-1]["end"] = l["end"]; out[-1]["text"] += ", " + text[0].lower() + text[1:] if not text.startswith("I ") else ", " + text
        else:
            out.append(dict(start=round(l["start"], 2), end=round(l["end"], 2), text=text))
    return out


def talk_lines(words):
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        gap = words[i + 1]["start"] - w["end"] if i + 1 < len(words) else 9
        txt = " ".join(x["word"] for x in cur)
        if gap > 0.35 or re.search(r"[.!?,]$", w["word"]) or len(cur) >= 4 or len(txt) >= 22:
            out.append(dict(start=round(cur[0]["start"], 2), end=round(w["end"] + min(0.25, gap), 2), text=txt.rstrip(",")))
            cur = []
    return out


def activity_end(paths, duration):
    """last moment anything is playing/singing (+0.6 s), so the export ends before the creator reaches for the phone"""
    import soundfile as sf
    last = 0
    for p in paths:
        x, sr = sf.read(p, always_2d=True); x = x.mean(1)
        h = int(0.25 * sr)
        db = np.array([20 * np.log10(np.sqrt(np.mean(x[i:i + h] ** 2)) + 1e-9) for i in range(0, len(x) - h, h)])
        act = np.nonzero(db > db.max() - 28)[0]
        if len(act): last = max(last, (act[-1] + 1) * 0.25)
    return round(min(duration - 0.5, last + 0.6), 2) if last else round(duration, 2)


def extras(p):
    """reference videos -> starting style; script -> caption words + the AI character's lines"""
    notes = []
    refs = [r for r in p.get("references") or [] if os.path.exists(r)]
    if refs:
        import reference
        progress("reference", 0.2, "Studying your reference videos")
        st = reference.style(sys.modules[__name__], refs)
        progress("reference", 1.0)
        if st:
            p["reference_style"] = st
            p["look"] = st["look"]
            if p["template"] in ("montage", "custom"):
                p["options"] = {**p.get("options", {}), **{k: st[k] for k in ("transition", "beat_sync", "music")}}
            notes.append(st["note"])
    if (p.get("script") or "").strip():
        import script, skit
        case = (lambda t: skit.caption_case(t, False)) if p["template"] in ("skit", "talk", "montage", "custom") else None
        n = 0
        if p["template"] in ("montage", "custom"):
            for c in p.get("clips", []):
                _, k = script.fix_captions(c.get("lines", []), p["script"], p.get("ai_name", ""), case); n += k
        else:
            _, n = script.fix_captions(p.get("lines", []), p["script"], p.get("ai_name", ""), case)
        if n: notes.append(f"{n} caption line{'s' if n > 1 else ''} corrected from your script.")
        if p["template"] == "skit":
            ai = script.ai_lines(p["script"], p.get("ai_name", ""))
            slots = [r for r in sorted(p.get("replies", []), key=lambda r: r["start"]) if r.get("mode", "reply") == "reply"]
            for r, t in zip(slots, ai): r["text"] = t
            if ai: notes.append(f"{min(len(ai), len(slots))} AI line{'s' if len(ai) > 1 else ''} filled from your script"
                                + (f" ({len(ai)} in the script, {len(slots)} pauses in the video)." if len(ai) != len(slots) else "."))
    if notes: p["ai_note"] = " ".join(([p["ai_note"]] if p.get("ai_note") else []) + notes)


def analyze(d):
    p = load(d)
    work = os.path.join(d, "work"); os.makedirs(work, exist_ok=True)
    if p["template"] in ("montage", "custom"):
        import montage
        montage.analyze(sys.modules[__name__], d, p)
        extras(p)
        p["analyzed"] = True
        save(d, p)
        emit("done", project=p)
        return
    progress("probe", 0.0, "Reading the video")
    info = probe(p["video"])
    if not info["has_audio"]: raise RuntimeError("This video has no sound track.")
    p.update(duration=info["duration"], width=info["width"], height=info["height"], fps=info["fps"])
    audio = os.path.join(work, "audio.wav")
    run(["ffmpeg", "-v", "error", "-y", "-i", p["video"], "-vn", "-ac", "1", "-ar", "44100", audio])
    progress("probe", 1.0)
    song = p.get("song") or {}
    if p["template"] == "skit":
        import skit
        skit.analyze(sys.modules[__name__], d, p)
        extras(p)
        p["analyzed"] = True
        save(d, p)
        emit("done", project=p)
        return
    if p["template"] == "cover":
        progress("separate", 0.02, "Separating voice and instrument (takes a few minutes)")
        run(SEPARATOR + [audio, "--model_file_dir", MODELS, "-m", SEP_MODEL, "--output_dir", work, "--output_format", "WAV"])
        for f in os.listdir(work):
            if "(Vocals)" in f: os.replace(os.path.join(work, f), os.path.join(work, "vocals.wav"))
            elif "(Instrumental)" in f: os.replace(os.path.join(work, f), os.path.join(work, "inst.wav"))
        progress("separate", 1.0)
        progress("transcribe", 0.0, "Listening for the lyrics")
        prompt = f"{song.get('title')}, {song.get('artist')}." if song.get("title") else None
        words = transcribe(os.path.join(work, "vocals.wav"), prompt)
        lines = cover_lines(words, phrases(os.path.join(work, "vocals.wav")))
        out = activity_end([os.path.join(work, "vocals.wav"), os.path.join(work, "inst.wav")], info["duration"])
    else:
        progress("transcribe", 0.0, "Transcribing")
        words = transcribe(audio)
        lines = talk_lines(words)
        out = activity_end([audio], info["duration"])
    progress("transcribe", 1.0)
    import llm
    p["ai_note"] = ""
    if llm.available() and lines:
        progress("ai", 0.1, "AI is checking the words")
        try:
            fixed = llm.fix_lines(p["template"], song, [l["text"] for l in lines])
            for l, t in zip(lines, fixed): l["text"] = t.strip() or l["text"]
            p["ai_note"] = "Words checked by AI."
        except llm.LLMError as e:
            p["ai_note"] = f"AI check skipped: {e}"
        progress("ai", 1.0)
    p["lines"] = lines
    p["trim"] = dict(**{"in": 0.0, "out": out, "fade_out": 1.8 if p["template"] == "cover" else 0.4})
    p.setdefault("look", "moody" if p["template"] == "cover" else "clean")
    extras(p)
    p["analyzed"] = True
    save(d, p)
    mix(d, p)
    emit("done", project=p)


# ---------------------------------------------------------------- audio
def mix(d, p, yt=False):
    import audio
    work = os.path.join(d, "work")
    out = os.path.join(work, "mix_yt.wav" if yt else "mix.wav")
    progress("mix", 0.1, "Polishing the sound")
    if p["template"] == "cover":
        lufs = audio.mix_cover(os.path.join(work, "vocals.wav"), os.path.join(work, "inst.wav"), out, yt=yt, inst_db=p.get("inst_db", -3.0))
    else:
        lufs = audio.clean_speech(os.path.join(work, "audio.wav"), out)
    progress("mix", 1.0, f"Sound at {lufs:.1f} LUFS")
    return out


# ---------------------------------------------------------------- video
def frames(path, a, b, w=W, h=H, fps=FPS):
    """frames scaled to cover w x h (centre crop), constant fps"""
    vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps}"
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", path, "-vf", vf,
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        buf = pr.stdout.read(n)
        if len(buf) < n: break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    pr.wait()


def zoom_frame(x, z, cy=0.34):
    import cv2
    h, w = x.shape[:2]
    M = np.float32([[z, 0, (1 - z) * w * 0.5], [0, z, (1 - z) * h * cy]])
    return cv2.warpAffine(x, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def still(d, look, t):
    from looks import Grader
    import cv2
    p = load(d)
    t = float(t)
    fr = next(frames(p["video"], t, t + 0.2), None)
    if fr is None: raise RuntimeError("Could not read a frame there.")
    out = Grader(look, W, H)(fr, 0)
    os.makedirs(os.path.join(d, "stills"), exist_ok=True)
    path = os.path.join(d, "stills", f"{look}.jpg")
    cv2.imwrite(path, cv2.cvtColor(cv2.resize(out, (360, 640), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    emit("done", path=path, look=look)


def render(d, target):
    p = load(d)
    if p["template"] in ("skit", "montage", "custom"):
        mod = __import__("skit" if p["template"] == "skit" else "montage")
        out = mod.render(sys.modules[__name__], d, p, target)
        emit("done", path=out, size=os.path.getsize(out)); return
    import cv2
    from looks import Grader
    import overlay
    p = load(d)
    yt = target == "yt"
    s = 2 if yt else 1
    a, b = p["trim"]["in"], p["trim"]["out"]
    dur = b - a
    nframes = int(dur * FPS)
    aud = mix(d, p, yt=yt)  # always re-mix: the instrument level may have changed
    grade = Grader(p.get("look", "moody"), W, H)
    items = overlay.build(p, s)
    os.makedirs(os.path.join(d, "exports"), exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", p.get("name", "reel")).strip("_") or "reel"
    out = os.path.join(d, "exports", f"{stem}_{'YouTube_4K' if yt else 'Instagram'}.mp4")
    tmp = out + ".video.mp4"
    ow, oh = W * s, H * s
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ow}x{oh}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium" if yt else "slow", "-crf", "16" if yt else "18", "-pix_fmt", "yuv420p", tmp],
                           stdin=subprocess.PIPE)
    fo = p["trim"].get("fade_out", 1.0)
    cover = p["template"] == "cover"
    progress("render", 0.0, "Rendering the video")
    t_last = time.time()
    for i, fr in enumerate(frames(p["video"], a, b)):
        t = i / FPS
        x = grade(fr, i)
        if cover: x = zoom_frame(x, 1.03 + 0.07 * t / max(dur, 1))
        if s != 1: x = cv2.resize(x, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
        x = np.ascontiguousarray(x)
        for it in items: it.draw(x, t)
        fade = min(1.0, t / 0.3, (dur - t) / fo if fo else 1.0)
        if fade < 1: x = (x.astype(np.float32) * max(0, fade)).astype(np.uint8)
        enc.stdin.write(x.tobytes())
        if time.time() - t_last > 0.5:
            progress("render", i / max(1, nframes)); t_last = time.time()
    enc.stdin.close(); enc.wait()
    progress("render", 1.0)
    progress("mux", 0.5, "Adding the sound")
    run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-ss", f"{a:.3f}", "-t", f"{dur:.3f}", "-i", aud, "-map", "0:v", "-map", "1:a",
         "-af", f"afade=t=in:d=0.15,afade=t=out:st={max(0, dur - fo):.3f}:d={fo:.3f}", "-c:v", "copy", "-c:a", "aac",
         "-b:a", "320k" if yt else "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart", out])
    os.remove(tmp)
    progress("mux", 1.0)
    emit("done", path=out, size=os.path.getsize(out))


# ---------------------------------------------------------------- caption
def caption(d, platform):
    import llm
    p = load(d)
    song = p.get("song") or {}
    texts = [l["text"] for l in p.get("lines", [])]
    if llm.available():
        try:
            emit("done", text=llm.post_caption(p["template"], song, texts, platform), ai=True); return
        except llm.LLMError as e:
            emit("log", msg=f"AI caption failed ({e}); using the template.")
    hook = texts[0] if texts else ""
    if p["template"] == "cover" and song.get("title"):
        t = song["title"]; ar = song.get("artist", "")
        tag = lambda x: "#" + re.sub(r"[^a-z0-9]", "", x.lower())
        body = (f"{hook}… 🥀\n\nMy cover of {t}{' by ' + ar if ar else ''}. Just me and my guitar, one take.\n\n"
                f"Should I sing the full version? Tell me in the comments 👇\n\n"
                f"Original song: {t}{', ' + ar if ar else ''}. All rights belong to the original artists.\n\n"
                f"{tag(t)} {tag(ar) if ar else ''} {tag(t + 'cover')} #acousticcover #guitarcover #singing #cover"
                + (" #shorts" if "YouTube" in platform else " #reels"))
        if "YouTube" in platform: body = f"Title: {t} - {ar} | Acoustic Cover\n\n" + body
    else:
        body = (f"{hook}…\n\n{' '.join(texts)[:180]}…\n\nWhat do you think? Tell me in the comments 👇\n\n"
                + ("#shorts" if "YouTube" in platform else "#reels"))
    emit("done", text=re.sub(r" +", " ", body), ai=False)


# ---------------------------------------------------------------- check
def check():
    have = {"ffmpeg": bool(shutil.which("ffmpeg"))}
    mods = {}
    for m in ("faster_whisper", "cv2", "pedalboard", "noisereduce", "pyloudnorm", "librosa", "anthropic", "PIL", "kokoro_onnx", "audio_separator.separator", "mediapipe", "av"):  # the separator module itself, so missing runtime deps show up
        try: __import__(m); mods[m] = True
        except Exception: mods[m] = False
    sep = mods["audio_separator.separator"]
    model = os.path.exists(os.path.join(MODELS, SEP_MODEL)) and os.path.exists(os.path.join(MODELS, "kokoro", "kokoro-v1.0.onnx"))
    ok = all(have.values()) and all(mods.values()) and sep and model
    emit("done", ok=ok, tools=have, modules=mods, separator=sep, separator_model=model, python=sys.executable)


def suggest_replies(d):
    import skit
    p = load(d)
    r = skit.suggest(p)
    emit("done", replies=r)


def voice(d, idx):
    """synthesize one AI reply for preview -> work/preview_<i>.wav"""
    import tts
    p = load(d)
    r = sorted(p.get("replies", []), key=lambda r: r["start"])[int(idx)]
    if not r.get("text", "").strip(): raise RuntimeError("This reply has no text yet.")
    out = os.path.join(d, "work", f"preview_{idx}.wav")
    secs = tts.synth(r["text"].strip(), out, MODELS, voice=p.get("ai_voice", "am_michael"), speed=1.1)
    room = r["end"] - r["start"]
    emit("done", path=out, seconds=round(secs, 2), room=round(room, 2), fits=secs <= room - 0.2)


def setup():
    """first run: download the models (speech recognition, voice/instrument separation, AI voice, face finder)"""
    import tts, urllib.request
    os.makedirs(MODELS, exist_ok=True)
    progress("models", 0.02, "Speech recognition model")
    from faster_whisper import WhisperModel
    WhisperModel("small", device="cpu", compute_type="int8", download_root=WHISPER_DIR)
    progress("models", 0.3, "AI voice")
    tts.ensure_model(MODELS, lambda f: progress("models", 0.3 + 0.2 * f, "AI voice"))
    progress("models", 0.5, "Face finder")
    seg = os.path.join(MODELS, "mediapipe", "selfie_multiclass_256x256.tflite")
    if not os.path.exists(seg):
        import skit
        os.makedirs(os.path.dirname(seg), exist_ok=True); urllib.request.urlretrieve(skit.SEG_URL, seg)
    progress("models", 0.55, "Voice / instrument separation (largest)")
    if not os.path.exists(os.path.join(MODELS, SEP_MODEL)):
        run(SEPARATOR + ["--download_model_only", "--model_file_dir", MODELS, "-m", SEP_MODEL])
    progress("models", 1.0, "Ready")
    emit("done", ok=True)


def llm_test():
    import llm
    emit("done", reply=llm.ask("Reply with the single word: ready"))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    try:
        if cmd == "check": check()
        elif cmd == "llm-test": llm_test()
        elif cmd == "analyze": analyze(args[0])
        elif cmd == "still": still(*args[:3])
        elif cmd == "render": render(args[0], args[1])
        elif cmd == "caption": caption(args[0], args[1] if len(args) > 1 else "Instagram")
        elif cmd == "suggest-replies": suggest_replies(args[0])
        elif cmd == "voice": voice(args[0], args[1])
        elif cmd == "setup": setup()
        else: raise RuntimeError(f"unknown command {cmd}")
    except Exception as e:
        import llm
        emit("error", msg=str(e), kind="llm" if isinstance(e, llm.LLMError) else "engine")
        sys.exit(1)
