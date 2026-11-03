"""Clips + transitions / Custom template: several clips -> trimmed, graded, captioned where someone speaks, joined
with transitions (whip, zoom, flash, crossfade, cut), optionally cut on the beat of the music; header + end card.
Project fields: clips[{path, in, out, lines[{start,end,text,skip}] (clip seconds), speech}], options{transition, music,
captions: bold|serif|off, beat_sync, punch_in, clip_sound}, look, header, endcard."""
import os, re, sys, subprocess, numpy as np, cv2, soundfile as sf

W, H, FPS, SR = 1080, 1920, 30, 48000
BEAT = 60 / 125
TRANS_FRAMES = {"whip": 8, "zoom": 8, "flash": 6, "crossfade": 10, "cut": 0}
DEFAULTS = {"transition": "whip", "music": "upbeat", "captions": "bold", "beat_sync": True, "punch_in": True, "clip_sound": True}


def analyze(E, d, p):
    import skit
    work = os.path.join(d, "work"); os.makedirs(work, exist_ok=True)
    clips = p.get("clips") or []
    if not clips: raise RuntimeError("Add at least one video.")
    for k, c in enumerate(clips):
        E.progress("probe", k / len(clips), f"Reading clip {k + 1}")
        info = E.probe(c["path"])
        c["duration"] = info["duration"]
        wav = os.path.join(work, f"clip{k}.wav")
        if info["has_audio"]:
            E.run(["ffmpeg", "-v", "error", "-y", "-i", c["path"], "-vn", "-ac", "1", "-ar", "44100", wav])
            sp = skit.speech_intervals(wav)
            x, sr = sf.read(wav, always_2d=True)
            loud = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
            sp = sp if loud > -45 else []
        else:
            sp = []
        c["speech"] = bool(sp)
        if sp:
            E.progress("transcribe", k / len(clips), f"Listening to clip {k + 1}")
            groups = [list(sp[0])]
            for a, b in sp[1:]:
                if a - groups[-1][1] >= 1.5: groups.append([a, b])
                else: groups[-1][1] = b
            words = skit.transcribe_groups(E, wav, groups, prompt=p.get("vocab") or None)
            c["lines"] = [l for g, ws in zip(groups, words) for l in skit.captions(ws)]
            c.setdefault("in", round(max(0.0, sp[0][0] - 0.25), 2)); c.setdefault("out", round(min(info["duration"], sp[-1][1] + 0.4), 2))
        else:
            c["lines"] = []
            c.setdefault("in", 0.0); c.setdefault("out", round(min(info["duration"], 4.0), 2))
    E.progress("transcribe", 1.0)
    p["options"] = {**DEFAULTS, **(p.get("options") or {})}
    p.setdefault("look", "clean")
    p.setdefault("header", "")
    p.setdefault("endcard", {"title": "", "subtitle": "", "cta": "", "url": ""})
    p["lines"] = []  # captions live per clip
    p["trim"] = {"in": 0.0, "out": 0.0, "fade_out": 0.0}
    p["duration"] = sum(c["out"] - c["in"] for c in clips)


def layout(p):
    """clip lengths (beat-snapped when music drives the edit) and output start times; transitions overlap clips"""
    o = {**DEFAULTS, **(p.get("options") or {})}
    n = TRANS_FRAMES.get(o["transition"], 0) / FPS
    out, t = [], 0.0
    for k, c in enumerate(p["clips"]):
        a, b = c["in"], c["out"]
        L = b - a
        if o["beat_sync"] and o["music"] != "none" and not c.get("speech"):
            L = max(2, round(L / BEAT)) * BEAT  # whole beats, so every cut lands on the beat
            b = min(c.get("duration", a + L), a + L)
        out.append(dict(k=k, a=a, b=b, len=b - a, at=t))
        t += (b - a) - (n if k < len(p["clips"]) - 1 else 0)
    return out, t, n


def transition(kind, A, B, i, n):
    """frame i of n while going from frame A (outgoing) to B (incoming)"""
    k = (i + 0.5) / n
    if kind == "crossfade":
        return cv2.addWeighted(A, 1 - k, B, k, 0)
    if kind == "flash":
        base = A if k < 0.5 else B
        f = 1 - abs(k - 0.5) * 2
        return np.clip(base.astype(np.float32) + 255 * 0.85 * f, 0, 255).astype(np.uint8)
    if kind == "zoom":
        img, z = (A, 1 + 0.35 * (k / 0.5) ** 2) if k < 0.5 else (B, 1 + 0.35 * ((1 - k) / 0.5) ** 2)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        r = int(2 + 18 * (1 - abs(k - 0.5) * 2))
        return cv2.blur(img, (r, r))
    if kind == "whip":
        e = k * k * (3 - 2 * k)
        shift = int(W * e)
        img = np.zeros_like(A)
        img[:, :W - shift] = A[:, shift:]
        img[:, W - shift:] = B[:, :shift]
        blur = int(4 + 60 * (1 - abs(k - 0.5) * 2))
        return cv2.blur(img, (blur, 1))
    return B


def render(E, d, p, target):
    import audio as A, skit
    from looks import Grader
    yt = target == "yt"; s = 2 if yt else 1
    OW, OH = int(W * s), int(H * s)
    work = os.path.join(d, "work"); os.makedirs(work, exist_ok=True)
    o = {**DEFAULTS, **(p.get("options") or {})}
    shots, total_clips, ov = layout(p)
    nT = TRANS_FRAMES.get(o["transition"], 0)
    ec = p.get("endcard") or {}
    has_end = any(ec.get(k) for k in ("title", "subtitle", "cta", "url"))
    end_at = total_clips + (0.3 if has_end else 0)
    total = end_at + (4.0 if has_end else 0)

    # ---- audio: clip sound (speech cleaned) + music ducked under speech
    E.progress("mix", 0.05, "Mixing")
    N = int(total * SR); mix = np.zeros((N, 2), np.float32); speech_mask = np.zeros(N, np.float32)
    for sh in shots:
        c = p["clips"][sh["k"]]
        wav = os.path.join(work, f"clip{sh['k']}.wav")
        if not os.path.exists(wav) or not o["clip_sound"] and not c.get("speech"): continue
        src = wav
        if c.get("speech"):
            src = os.path.join(work, f"clip{sh['k']}_clean.wav"); A.clean_speech(wav, src)
        x, sr0 = sf.read(src, always_2d=True)
        if sr0 != SR:
            import librosa; x = librosa.resample(x.T, orig_sr=sr0, target_sr=SR).T
        if x.shape[1] == 1: x = np.repeat(x, 2, 1)
        seg = x[int(sh["a"] * SR):int(sh["b"] * SR)].copy()
        if not c.get("speech"): seg *= 0.25 if o["music"] != "none" else 0.8
        f = min(int(0.03 * SR), len(seg) // 2)
        if f: seg[:f] *= np.linspace(0, 1, f)[:, None]; seg[-f:] *= np.linspace(1, 0, f)[:, None]
        at = int(sh["at"] * SR); n = min(len(seg), N - at)
        mix[at:at + n] += seg[:n]
        if c.get("speech"): speech_mask[at:at + n] = 1
    if o["music"] != "none":
        mfile = os.path.join(work, f"m_{o['music']}.wav"); here = os.path.dirname(os.path.abspath(__file__))
        if o["music"] == "comedic":
            subprocess.run([sys.executable, os.path.join(here, "music", "comedic.py"), f"{total}", f"{total + 1}", f"{total + 1}", f"{total + 1}", f"{end_at}", mfile], check=True, capture_output=True)
        else:
            subprocess.run([sys.executable, os.path.join(here, "music", "demo.py"), f"{total}", f"{total_clips}", f"{end_at}", mfile], check=True, capture_output=True)
        m, _ = sf.read(mfile, always_2d=True); n = min(len(m), N)
        k = int(0.25 * SR); duck = np.convolve(speech_mask, np.ones(k) / k, mode="same")[:n]
        vol = 0.55 - 0.42 * duck
        vol = np.where(np.arange(n) / SR > end_at - 0.05, 0.55, vol)
        mix[:n] += m[:n] * vol[:, None]
    final = os.path.join(work, "final_yt.wav" if yt else "final.wav")
    A._finish(mix.T.copy(), SR, final)

    # ---- video
    grade = Grader(p.get("look", "clean"), W, H)
    fcap = skit.font("Poppins-Bold.ttf", int(56 * s)) if o["captions"] != "serif" else None
    fser = skit.font("CormorantGaramond[wght].ttf", int(56 * s))
    try: fser.set_variation_by_axes([600])
    except Exception: pass
    caps = []
    if o["captions"] != "off":
        for sh in shots:
            for l in p["clips"][sh["k"]].get("lines", []):
                if l.get("skip") or not l["text"].strip() or l["end"] <= sh["a"] or l["start"] >= sh["b"]: continue
                a = sh["at"] + max(l["start"], sh["a"]) - sh["a"]; b = sh["at"] + min(l["end"], sh["b"]) - sh["a"]
                if o["captions"] == "serif":
                    spr = np.asarray(skit.text_block(l["text"], fser, 900 * s, fill=(244, 242, 238, 255), shadow=int(10 * s)))
                else:
                    spr = np.asarray(skit.text_block(l["text"], fcap, 940 * s, stroke=int(5 * s), accent=(255, 210, 63, 255)))
                caps.append((a, b, spr))
    hdr = np.asarray(skit.header_box(p["header"], s)) if p.get("header") else None
    endimg = skit.end_card(p, s) if has_end else None
    tmp = os.path.join(work, "video.mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "medium" if yt else "slow", "-crf", "16" if yt else "18", "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
    nframes = int(total * FPS); fi = 0
    E.progress("render", 0.0, "Rendering")

    def finish_frame(img, t):
        nonlocal fi
        if s != 1: img = cv2.resize(img, (OW, OH), interpolation=cv2.INTER_LANCZOS4)
        img = np.ascontiguousarray(img)
        if hdr is not None: skit.blend(img, hdr, (OW - hdr.shape[1]) // 2, int(248 * s))
        for a, b, spr in caps:
            if a <= t < b:
                y = int((1300 if o["captions"] != "serif" else 1470) * s)
                skit.blend(img, spr, (OW - spr.shape[1]) // 2, y, alpha=min(1.0, (t - a) / 0.1, (b - t) / 0.1))
        if fi % 15 == 0: E.progress("render", fi / nframes)
        enc.stdin.write(img.tobytes()); fi += 1

    held = []  # the outgoing clip's last frames, waiting for the transition into the next clip
    for j, sh in enumerate(shots):
        c = p["clips"][sh["k"]]
        n = round(sh["len"] * FPS)
        z = 1.0 if (not o["punch_in"] or j % 2 == 0) else 1.12
        frames = []
        for i, fr in enumerate(E.frames(c["path"], sh["a"], sh["b"] + 0.1)):
            if i >= n: break
            img = grade(fr, fi + i)
            zz = z + 0.04 * i / max(1, n - 1)
            if zz != 1.0:
                M = cv2.getRotationMatrix2D((W / 2, H * 0.45), 0, zz); img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            frames.append(img)
        if not frames: continue
        while len(frames) < n: frames.append(frames[-1])
        start = 0
        if held:
            for i in range(len(held)):
                finish_frame(transition(o["transition"], held[i], frames[min(i, len(frames) - 1)], i, len(held)), sh["at"] + i / FPS)
            start = len(held); held = []
        last = j == len(shots) - 1
        keep = n if last or nT == 0 else n - nT
        for i in range(start, keep):
            finish_frame(frames[i], sh["at"] + i / FPS)
        if not last and nT: held = frames[keep:]
    while fi < round(end_at * FPS): finish_frame(np.zeros((H, W, 3), np.uint8), 0)
    if endimg is not None:
        for k in range(round(4.0 * FPS)):
            img = (endimg.astype(np.float32) * min(1.0, k / 8)).astype(np.uint8)
            if fi % 15 == 0: E.progress("render", fi / nframes)
            enc.stdin.write(np.ascontiguousarray(img).tobytes()); fi += 1
    enc.stdin.close(); enc.wait()
    E.progress("render", 1.0)
    os.makedirs(os.path.join(d, "exports"), exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", p.get("name", "reel")).strip("_") or "reel"
    out = os.path.join(d, "exports", f"{stem}_{'YouTube_4K' if yt else 'Instagram'}.mp4")
    E.progress("mux", 0.5, "Adding the sound")
    E.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-i", final, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
           "-b:a", "320k" if yt else "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart", out])
    os.remove(tmp)
    E.progress("mux", 1.0)
    return out
