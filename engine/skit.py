"""AI skit template: one continuous take where the creator talks to an AI on the laptop and leaves pauses for it to answer.
analyze: speech/pause map, captions (creator only), reply slots at the pauses, face position, trim.
render:  AI voice (Kokoro, laptop speaker) in each reply slot; dead air cut with alternating punch-ins; optional product
         demo from a screen recording at one slot (beat-cut, punch-in shots, light sweeps, text); comedic score + demo
         track; POV header; end card with a call to action.
Project fields: replies[{start,end,text,mode: reply|demo|cut}], ai_voice, header, demo{video,texts[5],starts?},
endcard{title,subtitle,cta,url}, caption_lowercase, face{cx,cy,bottom}."""
import os, re, sys, json, subprocess, tempfile, numpy as np, cv2, soundfile as sf
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "fonts")
W, H, FPS, SR = 1080, 1920, 30, 48000
BEAT = 60 / 125
ACCENT = (46, 230, 182)


def font(name, size): return ImageFont.truetype(os.path.join(FONTS, name), size)


# ---------------------------------------------------------------- analysis
def speech_intervals(wav):
    x, sr = sf.read(wav, always_2d=True); x = x.mean(1)
    h = int(0.1 * sr)
    db = np.array([20 * np.log10(np.sqrt(np.mean(x[i:i + h] ** 2)) + 1e-9) for i in range(0, len(x) - h, h)])
    on = db > db.max() - 30
    out, s = [], None
    for i, o in enumerate(list(on) + [False]):
        if o and s is None: s = i
        if not o and s is not None:
            if i - s >= 2: out.append([s / 10, i / 10])
            s = None
    merged = []
    for a, b in out:
        if merged and a - merged[-1][1] < 0.45: merged[-1][1] = b
        else: merged.append([a, b])
    return merged


SEG_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"


def find_face(E, video, duration):
    """median face box over a few frames (1080x1920 output coords) from the selfie segmenter's face-skin class:
    where to punch in and where the captions go (just under the chin)"""
    import mediapipe as mp, urllib.request
    from mediapipe.tasks.python import vision, BaseOptions
    path = os.path.join(E.MODELS, "mediapipe", "selfie_multiclass_256x256.tflite")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True); urllib.request.urlretrieve(SEG_URL, path)
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=path), output_confidence_masks=True))
    boxes = []
    for t in np.linspace(duration * 0.1, duration * 0.85, 9):
        fr = next(E.frames(video, t, t + 0.1), None)
        if fr is None: continue
        small = np.ascontiguousarray(cv2.resize(fr, (288, 512), interpolation=cv2.INTER_AREA))
        conf = [np.squeeze(c.numpy_view()) for c in seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=small)).confidence_masks]
        face = cv2.resize(conf[3], (288, 512)) > 0.5
        n, lab, st, _ = cv2.connectedComponentsWithStats(face.astype(np.uint8))
        if n < 2: continue
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if st[k, cv2.CC_STAT_AREA] < 300: continue
        x, y, w, h = st[k, :4] * (W / 288)
        boxes.append((x, y, w, h))
    if not boxes: return dict(cx=0.5, cy=0.42, bottom=0.55)
    x, y, w, h = np.median(np.array(boxes, np.float32), 0)
    return dict(cx=round(float(x + w / 2) / W, 3), cy=round(float(y + h / 2) / H, 3), bottom=round(float(y + h) / H, 3))


def caption_case(text, first_in_sentence):
    """reference style: lowercase, but keep acronyms (AI, GPT) and names Whisper capitalised mid-sentence"""
    out = []
    for i, w in enumerate(text.split()):
        core = re.sub(r"[^A-Za-z']", "", w)
        if len(core) > 1 and core.isupper(): out.append(w)
        elif core[:1].isupper() and not (i == 0 and first_in_sentence) and core not in ("I", "I'm", "I've", "I'll", "I'd"): out.append(w)
        else: out.append(w.lower())
    return " ".join(out)


def captions(words):
    """2-6 word chunks; break at punctuation or a pause, never leave a 1-2 word tail before the next punctuation"""
    ws = []
    for w in words:  # whisper splits hyphenated words: "off" "-pitch"
        if ws and w["word"].startswith("-"): ws[-1] = dict(ws[-1], word=ws[-1]["word"] + w["word"], end=w["end"])
        else: ws.append(dict(w))
    out, cur, new_sentence, i = [], [], True, 0
    punct = lambda w: re.search(r"[.!?,]$", w["word"])
    while i < len(ws):
        w = ws[i]; cur.append(w)
        gap = ws[i + 1]["start"] - w["end"] if i + 1 < len(ws) else 9
        txt = " ".join(x["word"] for x in cur)
        brk = gap > 0.35 or punct(w)
        if not brk and (len(cur) >= 5 or len(txt) >= 26):
            # look ahead: finish the phrase if it ends within 2 words and stays short
            ahead = ws[i + 1:i + 3]
            k = next((j for j, x in enumerate(ahead) if punct(x)), None)
            if k is not None and len(txt) + sum(len(x["word"]) + 1 for x in ahead[:k + 1]) <= 36:
                cur += ahead[:k + 1]; i += k + 1; w = cur[-1]
                gap = ws[i + 1]["start"] - w["end"] if i + 1 < len(ws) else 9
            brk = True
        if brk:
            txt = " ".join(x["word"] for x in cur)
            out.append(dict(start=round(cur[0]["start"], 2), end=round(w["end"] + min(0.3, gap), 2), text=caption_case(txt, new_sentence)))
            new_sentence = bool(re.search(r"[.!?]$", w["word"])); cur = []
        i += 1
    return out


SLOT_MIN = 1.8  # a pause this long is left for the AI to answer


def transcribe_groups(E, wav, groups, prompt=None):
    """transcribe each stretch of speech on its own (one pass over a whole skit take drifts and mis-hears)"""
    from faster_whisper import WhisperModel
    import librosa
    x, sr = sf.read(wav, always_2d=True)
    y = librosa.resample(x.mean(1).astype(np.float32), orig_sr=sr, target_sr=16000)
    m = WhisperModel("small", device="cpu", compute_type="int8", download_root=E.WHISPER_DIR)
    out = []
    for gi, (a, b) in enumerate(groups):
        a0 = max(0.0, a - 0.3)
        segs, _ = m.transcribe(y[int(a0 * 16000):int((b + 0.3) * 16000)], language="en", word_timestamps=True,
                               condition_on_previous_text=False, temperature=0, vad_filter=False, initial_prompt=prompt)
        ws = [dict(start=float(w.start) + a0, end=float(w.end) + a0, word=w.word.strip()) for sg in segs for w in sg.words or []]
        out.append([w for w in ws if not re.fullmatch(r"(uh|um|-?uh|-)[-,.]*", w["word"].lower())])
        E.progress("transcribe", (gi + 1) / len(groups))
    return out


def analyze(E, d, p):
    work = os.path.join(d, "work")
    audio = os.path.join(work, "audio.wav")
    speech = speech_intervals(audio)
    if not speech: raise RuntimeError("No speech found in this video.")
    # a short sound long after the last line is the creator reaching for the phone, not part of the skit
    while len(speech) > 1 and speech[-1][1] - speech[-1][0] < 1.0 and speech[-1][0] - speech[-2][1] > 2.0:
        speech.pop()
    groups = [list(speech[0])]
    for a, b in speech[1:]:
        if a - groups[-1][1] >= SLOT_MIN: groups.append([a, b])
        else: groups[-1][1] = b
    E.progress("transcribe", 0.0, "Listening to your take")
    words = transcribe_groups(E, audio, groups, prompt=p.get("vocab") or None)
    caps = []
    for g, ws in zip(groups, words):
        for c in captions(ws):  # captions never cross a pause; clamp to the spoken stretch
            c["start"] = round(max(c["start"], g[0] - 0.05), 2); c["end"] = round(min(c["end"], g[1] + 0.25), 2)
            caps.append(c)
    slots = [dict(start=round(a[1] + 0.1, 2), end=round(b[0] - 0.1, 2), text="", mode="reply") for a, b in zip(groups, groups[1:])]
    # pauses inside a spoken stretch: tightened to ~0.35 s at render
    p["pauses"] = [[round(a[1] + 0.15, 2), round(b[0] - 0.15, 2)] for a, b in zip(speech, speech[1:]) if 0.8 <= b[0] - a[1] < SLOT_MIN]
    p["replies"] = slots
    p["lines"] = caps
    t_in = max(0.0, groups[0][0] - 0.3)
    t_out = min(p["duration"] - 0.4, groups[-1][1] + 1.6)  # hold the last reaction a beat
    p["trim"] = {"in": round(t_in, 2), "out": round(t_out, 2), "fade_out": 0.0}
    E.progress("face", 0.3, "Finding your face")
    p["face"] = find_face(E, p["video"], p["duration"])
    p.setdefault("ai_voice", "am_michael")
    p.setdefault("ai_name", "the AI")
    p.setdefault("header", "")
    p.setdefault("endcard", {"title": "", "subtitle": "", "cta": "", "url": ""})
    p.setdefault("demo", {"video": "", "texts": ["", "", "", "", ""]})
    p.setdefault("look", "daylight")
    import llm
    if llm.available() and caps:
        E.progress("ai", 0.1, "AI is checking your words")
        try:
            fixed = llm.fix_lines("talk", {}, [c["text"] for c in caps])
            for c, t in zip(caps, fixed): c["text"] = caption_case(t.strip(), False) if t.strip() else c["text"]
            if slots:
                for sl, t in zip(p["replies"], suggest(p)): sl["text"] = t
            p["ai_note"] = "AI checked your words and drafted the replies. Edit them in the Replies tab."
        except llm.LLMError as e:
            p["ai_note"] = f"AI step skipped: {e}"
        E.progress("ai", 1.0)


def suggest(p):
    import llm
    script = []
    slot_i = 0
    events = [(l["start"], "me", l["text"]) for l in p["lines"]] + [(s["start"], "slot", s) for s in p["replies"]]
    for t, kind, x in sorted(events, key=lambda e: e[0]):
        if kind == "me": script.append(f"ME: {x}")
        else:
            slot_i += 1; script.append(f"[SLOT {slot_i}: {x['end'] - x['start']:.1f}s]")
    schema = {"type": "object", "properties": {"replies": {"type": "array", "items": {"type": "string"}}}, "required": ["replies"], "additionalProperties": False}
    prompt = (f"This is a short comedy skit video. The creator talks to {p.get('ai_name') or 'an AI assistant'} on their laptop and leaves "
              f"pauses (SLOTs) where the AI answers out loud. Write what the AI says in each slot: short, natural, a little deadpan, "
              f"funny where it fits, and it must fit the slot length at about 2.7 words per second. Return exactly {slot_i} replies "
              f"in order. If a slot is clearly not meant for the AI (e.g. the creator is typing), return an empty string for it.\n\n"
              + "\n".join(script))
    out = llm.ask(prompt, schema=schema).get("replies", [])
    return (out + [""] * slot_i)[:slot_i]


# ---------------------------------------------------------------- timeline
def plan(p, reply_len, demo_len):
    """-> list of segments {kind: take|demo, a, b (take seconds) | len, zoom: (z0, z1), out: start in output}"""
    t_in, t_out = p["trim"]["in"], p["trim"]["out"]
    segs, cur = [], t_in
    voice_at = []  # (output time, reply index)
    for i, s in enumerate(sorted(p.get("replies", []), key=lambda s: s["start"])):
        if s["start"] < cur or s["start"] > t_out: continue
        mode = s.get("mode", "reply") if (s.get("text") or s.get("mode") == "demo") else "cut"
        if mode == "reply":
            keep = min(s["end"], s["start"] + 0.15 + reply_len.get(i, 0) + 0.35)
            segs.append(["take", cur, keep]); voice_at.append((cur, s["start"] + 0.15, i))
        elif mode == "demo":
            segs.append(["take", cur, s["start"] + 0.3]); segs.append(["demo", 0, demo_len])
        else:
            segs.append(["take", cur, s["start"] + 0.4])
        cur = max(segs[-1][2] if segs[-1][0] == "take" else s["end"], s["end"] - 0.25)
    segs.append(["take", cur, t_out])
    # remove skipped lines and dead air inside spoken stretches
    holes = [(l["start"] - 0.05, l["end"]) for l in p.get("lines", []) if l.get("skip")]
    holes += [(a, b) for a, b in p.get("pauses", [])]
    for ha, hb in sorted(holes):
        nxt = []
        for sg in segs:
            if sg[0] == "take" and sg[1] < ha and hb < sg[2]: nxt += [["take", sg[1], ha], ["take", hb, sg[2]]]
            elif sg[0] == "take" and ha <= sg[1] < hb < sg[2]: nxt.append(["take", hb, sg[2]])
            elif sg[0] == "take" and sg[1] < ha < sg[2] <= hb: nxt.append(["take", sg[1], ha])
            elif sg[0] == "take" and ha <= sg[1] and sg[2] <= hb: continue
            else: nxt.append(sg)
        segs = nxt
    out, t, alt = [], 0.0, 0
    for kind, a, b in segs:
        if kind == "take" and b - a < 0.2: continue
        if kind == "take" and out and out[-1]["kind"] == "take" and abs(out[-1]["b"] - a) < 1e-3:
            out[-1]["b"] = b; t += b - a; continue  # contiguous: no cut
        seg = dict(kind=kind, a=a, b=b, out=t)
        if kind == "take":
            z = 1.0 if alt % 2 == 0 else 1.13
            seg["zoom"] = (z, z + 0.03); alt += 1
            t += b - a
        else:
            seg["len"] = b; t += b
        out.append(seg)
    def to_out(src):
        for s in out:
            if s["kind"] == "take" and s["a"] - 1e-3 <= src <= s["b"] + 1e-3: return s["out"] + src - s["a"]
        return None
    voices = [(to_out(at), i) for _, at, i in voice_at if to_out(at) is not None]
    return out, t, to_out, voices


# ---------------------------------------------------------------- drawing
def rich_line(text, f, fill, accent, stroke=0):
    """draw text where *words* are accented -> RGBA image"""
    parts = re.split(r"(\*[^*]+\*)", text)
    runs = [(x.strip("*"), accent if x.startswith("*") else fill) for x in parts if x]
    w = int(sum(f.getlength(r) for r, _ in runs)) + 2 * stroke + 4
    a, dsc = f.getmetrics()
    im = Image.new("RGBA", (w, a + dsc + 2 * stroke + 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(im); x = stroke + 2
    for r, c in runs:
        d.text((x, stroke + 2), r, font=f, fill=c, stroke_width=stroke, stroke_fill=(0, 0, 0, 255)); x += f.getlength(r)
    return im


def text_block(text, f, maxw, fill=(255, 255, 255, 255), accent=(255, 210, 63, 255), stroke=0, shadow=0, gap=6):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if f.getlength(t.replace("*", "")) <= maxw or not cur: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    # keep *accent* spans balanced per line
    fixed, open_ = [], False
    for l in lines:
        l2 = ("*" if open_ else "") + l
        if l2.count("*") % 2: l2 += "*"; open_ = True
        else: open_ = False
        fixed.append(l2)
    ims = [rich_line(l, f, fill, accent, stroke) for l in fixed]
    bw = max(i.width for i in ims); bh = sum(i.height for i in ims) + gap * (len(ims) - 1)
    pad = shadow * 2
    im = Image.new("RGBA", (bw + 2 * pad, bh + 2 * pad), (0, 0, 0, 0))
    y = pad
    for i in ims: im.alpha_composite(i, ((im.width - i.width) // 2, y)); y += i.height + gap
    if shadow:
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0)); sh.putalpha(im.getchannel("A").point(lambda v: int(v * 0.6)))
        sh = sh.filter(ImageFilter.GaussianBlur(shadow)); sh.alpha_composite(im); im = sh
    return im


def header_box(text, s):
    f = font("Poppins-SemiBold.ttf", int(37 * s))
    tw = int(f.getlength(text)); a, dsc = f.getmetrics()
    pw, ph = int(28 * s), int(16 * s)
    im = Image.new("RGBA", (tw + 2 * pw + int(40 * s), a + dsc + 2 * ph + int(40 * s)), (0, 0, 0, 0))
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((int(20 * s), int(26 * s), im.width - int(20 * s), im.height - int(14 * s)), int(18 * s), fill=(0, 0, 0, 70))
    sh = sh.filter(ImageFilter.GaussianBlur(14 * s))
    d = ImageDraw.Draw(sh)
    d.rounded_rectangle((int(20 * s), int(20 * s), im.width - int(20 * s), im.height - int(20 * s)), int(18 * s), fill=(255, 255, 255, 255))
    d.text((int(20 * s) + pw, int(20 * s) + ph), text, font=f, fill=(17, 17, 17, 255))
    return sh


def end_card(p, s):
    e = p.get("endcard") or {}
    im = Image.new("RGB", (int(W * s), int(H * s)), (5, 11, 10))
    yy, xx = np.mgrid[0:im.height, 0:im.width].astype(np.float32)
    r = np.sqrt(((xx - im.width / 2) / (im.width * 0.9)) ** 2 + ((yy - im.height * 0.4) / (im.height * 0.55)) ** 2)
    k = np.clip(1 - r, 0, 1)[..., None]
    bg = (np.float32([5, 11, 10]) * (1 - k) + np.float32([15, 58, 51]) * k)
    im = Image.fromarray(bg.astype(np.uint8)).convert("RGBA")
    blocks = []
    if e.get("title"): blocks.append(text_block(e["title"], font("Poppins-Bold.ttf", int(118 * s)), 960 * s, accent=ACCENT + (255,)))
    if e.get("subtitle"): blocks.append(text_block(e["subtitle"], font("Poppins-SemiBold.ttf", int(46 * s)), 940 * s, fill=(207, 233, 226, 255)))
    if e.get("cta"):
        f = font("Poppins-Bold.ttf", int(44 * s)); tw = int(f.getlength(e["cta"])); a, dsc = f.getmetrics()
        pill = Image.new("RGBA", (tw + int(72 * s), a + dsc + int(32 * s)), (0, 0, 0, 0)); d = ImageDraw.Draw(pill)
        d.rounded_rectangle((0, 0, pill.width - 1, pill.height - 1), pill.height // 2, fill=ACCENT + (255,))
        d.text((int(36 * s), int(16 * s)), e["cta"], font=f, fill=(4, 33, 27, 255)); blocks.append(pill)
    if e.get("url"): blocks.append(text_block(e["url"], font("Poppins-SemiBold.ttf", int(30 * s)), 900 * s, fill=(143, 179, 170, 255)))
    gaps = [int(10 * s), int(60 * s), int(24 * s)]
    total = sum(b.height for b in blocks) + sum(gaps[:max(0, len(blocks) - 1)])
    y = int(im.height * 0.47 - total / 2)
    for i, b in enumerate(blocks):
        im.alpha_composite(b, ((im.width - b.width) // 2, y)); y += b.height + (gaps[i] if i < len(gaps) else 0)
    return np.asarray(im.convert("RGB"))


def blend(frame, sprite, x, y, alpha=1.0):
    h, w = sprite.shape[:2]; H_, W_ = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W_, x + w), min(H_, y + h)
    if x1 <= x0 or y1 <= y0: return
    sp = sprite[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = sp[..., 3:4] / 255 * alpha
    fr = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (fr * (1 - a) + sp[..., :3] * a).astype(np.uint8)


# ---------------------------------------------------------------- demo (screen recording -> 9:16)
class Demo:
    def __init__(self, E, path, texts, starts, s, zoom=0.32, zooms=None):
        self.E, self.path, self.s, self.zoom, self.zooms = E, path, s, zoom, list(zooms or [])
        info = E.probe(path); self.SW, self.SH = info["width"], info["height"]; dur = info["duration"]
        lens = [4, 4, 6, 4, 4]
        auto = list(np.linspace(dur * 0.04, dur * 0.82, 5))
        starts = [st if st is not None else auto[k] for k, st in enumerate((list(starts or []) + [None] * 5)[:5])]
        self.shots = [(min(st, max(0, dur - l * BEAT - 0.1)), l * BEAT) for st, l in zip(starts, lens)]
        self.len = sum(round(l * FPS) for _, l in self.shots) / FPS
        self.texts = (list(texts or []) + [""] * 5)[:5]
        # content column: where the UI is (columns that change across the recording)
        fr = [f for f in self._frames(dur * 0.3, 0.1)][:1] + [f for f in self._frames(dur * 0.7, 0.1)][:1]
        g = np.mean([cv2.cvtColor(cv2.resize(f, (self.SW // 4, self.SH // 4)), cv2.COLOR_RGB2GRAY).astype(np.float32) for f in fr], 0)
        prof = g.std(0); act = np.nonzero(prof > prof.max() * 0.35)[0]
        x0, x1 = (act.min() * 4, act.max() * 4) if len(act) else (0, self.SW)
        self.cx = (x0 + x1) / 2
        self.w0 = min(self.SW, max(self.SH * 0.55, (x1 - x0) * 0.9), self.SH / 1.3)

    def fit(self, st, ln, k):
        """crop for one shot: centred on where the UI detail is; width = demo zoom (share of the screen width)"""
        frs = list(self._frames(st, 0.1))[:1] + list(self._frames(st + ln * 0.8, 0.1))[:1]
        g = np.mean([cv2.Canny(cv2.cvtColor(cv2.resize(f, (self.SW // 4, self.SH // 4)), cv2.COLOR_RGB2GRAY), 80, 200).astype(np.float32) for f in frs], 0)
        cols, rows = g.mean(0), g.mean(1)
        cols[:int(len(cols) * 0.12)] = 0  # app sidebar
        zoom = (self.zooms[k] if k < len(self.zooms) and self.zooms[k] else self.zoom)
        r = 1.3
        w = min(self.SW * zoom, self.SH / r)
        if cols.sum() < 1e-3: return self.SW / 2, self.SH / 2, w, r
        cx = np.searchsorted(np.cumsum(cols) / cols.sum(), 0.5) * 4
        cy = np.searchsorted(np.cumsum(rows) / max(rows.sum(), 1e-3), 0.5) * 4
        return float(np.clip(cx, w / 2, self.SW - w / 2)), float(cy), float(w), r

    def _frames(self, a, d):
        pr = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{d:.3f}", "-i", self.path, "-vf", f"fps={FPS}",
                               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        n = self.SW * self.SH * 3
        while True:
            b = pr.stdout.read(n)
            if len(b) < n: break
            yield np.frombuffer(b, np.uint8).reshape(self.SH, self.SW, 3)
        pr.wait()

    def frames(self):
        s = self.s; OW, OH = int(W * s), int(H * s)
        bw = int(940 * s); cy_box = int(1060 * s); rad = int(28 * s)
        for k, (st, ln) in enumerate(self.shots):
            n = round(ln * FPS)
            text = self.texts[k]
            cx, cy0, w0, r = self.fit(st, ln, k)
            bh = int(bw * r)
            mask = np.zeros((bh, bw), np.uint8); cv2.rectangle(mask, (0, 0), (bw - 1, bh - 1), 255, -1)
            for cx_, cy_ in ((rad, rad), (bw - rad, rad), (rad, bh - rad), (bw - rad, bh - rad)):
                cv2.rectangle(mask, (cx_ - rad if cx_ < bw / 2 else cx_, cy_ - rad if cy_ < bh / 2 else cy_),
                              (cx_ if cx_ < bw / 2 else cx_ + rad, cy_ if cy_ < bh / 2 else cy_ + rad), 0, -1)
                cv2.circle(mask, (cx_, cy_), rad, 255, -1)
            maskf = mask.astype(np.float32) / 255
            tsp = np.asarray(text_block(text, font("Poppins-Bold.ttf", int(54 * s)), 960 * s, accent=ACCENT + (255,), shadow=int(12 * s))) if text else None
            for i, fr in enumerate(self._frames(st, ln + 0.2)):
                if i >= n: break
                kk = i / max(1, n - 1); e = kk * kk * (3 - 2 * kk)
                w = w0 * (1 - 0.18 * e); h = w * r
                if h > self.SH: h = self.SH; w = h / r
                cy = cy0 + self.SH * 0.05 * e
                x0 = np.clip(cx - w / 2, 0, self.SW - w); y0 = np.clip(cy - h / 2, 0, self.SH - h)
                M = np.float32([[bw / w, 0, -x0 * bw / w], [0, bh / h, -y0 * bh / h]])
                box = cv2.warpAffine(fr, M, (bw, bh), flags=cv2.INTER_AREA).astype(np.float32)
                sc = max(OW / self.SW, OH / self.SH)
                bg = cv2.resize(fr, (int(self.SW * sc) + 1, int(self.SH * sc) + 1), interpolation=cv2.INTER_AREA)
                ox, oy = (bg.shape[1] - OW) // 2, (bg.shape[0] - OH) // 2
                bg = cv2.GaussianBlur(bg[oy:oy + OH, ox:ox + OW], (0, 0), 40 * s).astype(np.float32) * 0.45
                ent = min(1.0, i / 7); ee = 1 - (1 - ent) ** 3  # punch-in on the beat
                sc2 = 0.88 + 0.12 * ee
                bw2, bh2 = int(bw * sc2), int(bh * sc2)
                b2 = cv2.resize(box, (bw2, bh2)); m2 = cv2.resize(maskf, (bw2, bh2))
                bx0 = (OW - bw2) // 2; by0 = int(cy_box - bh2 / 2 + 110 * s * (1 - ee))
                shadow = np.zeros((OH, OW), np.float32)
                y_a, y_b = max(0, by0 + int(18 * s)), min(OH, by0 + int(18 * s) + bh2)
                shadow[y_a:y_b, bx0:bx0 + bw2] = m2[:y_b - y_a]
                bg *= (1 - 0.55 * cv2.GaussianBlur(shadow, (0, 0), 24 * s))[..., None]
                ya, yb = max(0, by0), min(OH, by0 + bh2)
                reg = bg[ya:yb, bx0:bx0 + bw2]; mm = m2[ya - by0:yb - by0][..., None]
                bg[ya:yb, bx0:bx0 + bw2] = reg * (1 - mm) + b2[ya - by0:yb - by0] * mm
                if 3 <= i < 19:  # light sweep across the glass
                    yy, xx = np.mgrid[0:yb - ya, 0:bw2].astype(np.float32)
                    pos = -bh2 * 0.6 + (bw2 + bh2 * 1.2) * (i - 3) / 16
                    bg[ya:yb, bx0:bx0 + bw2] += (np.exp(-(((xx + yy * 0.6) - pos) / (55 * s)) ** 2) * mm[..., 0] * 70)[..., None] * np.float32([0.75, 1, 0.95])
                if i < 3: bg += 0.32 * (1 - i / 3) * 255  # flash on the cut
                out = np.clip(bg, 0, 255).astype(np.uint8)
                if tsp is not None:
                    p_ = min(1.0, i / 6); pe = 1 - (1 - p_) ** 3
                    blend(out, tsp, (OW - tsp.shape[1]) // 2, int(250 * s + 30 * s * (1 - pe)), alpha=pe * min(1.0, (n - i) / 4))
                yield out


# ---------------------------------------------------------------- render
def render(E, d, p, target):
    import audio as A, tts
    from looks import Grader
    yt = target == "yt"; s = 2 if yt else 1
    work = os.path.join(d, "work"); os.makedirs(work, exist_ok=True)
    # 1) AI voice lines
    E.progress("voice", 0.0, "Recording the AI's lines")
    reps = sorted(p.get("replies", []), key=lambda r: r["start"])
    reply_len, voice_files = {}, {}
    todo = [i for i, r in enumerate(reps) if r.get("text", "").strip() and r.get("mode", "reply") == "reply"]
    for n, i in enumerate(todo):
        f = os.path.join(work, f"ai_{i}.wav")
        room = reps[i]["end"] - reps[i]["start"] - 0.3
        sp = 1.1
        reply_len[i] = tts.synth(reps[i]["text"].strip(), f, E.MODELS, voice=p.get("ai_voice", "am_michael"), speed=sp)
        if reply_len[i] > room:  # too long for the pause he left: speak a little faster (the app warns before this)
            sp = min(1.35, sp * reply_len[i] / room)
            reply_len[i] = tts.synth(reps[i]["text"].strip(), f, E.MODELS, voice=p.get("ai_voice", "am_michael"), speed=sp)
        voice_files[i] = f
        E.progress("voice", (n + 1) / max(1, len(todo)))
    p2 = dict(p); p2["replies"] = reps
    demo = None
    if any(r.get("mode") == "demo" for r in reps) and p.get("demo", {}).get("video"):
        dm = p["demo"]
        demo = Demo(E, dm["video"], dm.get("texts"), dm.get("starts"), s, zoom=dm.get("zoom", 0.32), zooms=dm.get("zooms"))
    segs, total_take, to_out, voices = plan(p2, reply_len, demo.len if demo else 0)
    ec = p.get("endcard") or {}
    has_end = any(ec.get(k) for k in ("title", "subtitle", "cta", "url"))
    black = total_take
    end_at = black + (0.45 if has_end else 0)
    total = end_at + (4.0 if has_end else 0.3)
    demo_seg = next((x for x in segs if x["kind"] == "demo"), None)

    # 2) audio
    E.progress("mix", 0.1, "Mixing")
    speech = os.path.join(work, "speech.wav")
    A.clean_speech(os.path.join(work, "audio.wav"), speech)
    sp, sr0 = sf.read(speech, always_2d=True)
    if sr0 != SR:
        import librosa; sp = librosa.resample(sp.T, orig_sr=sr0, target_sr=SR).T
    N = int(total * SR); mix = np.zeros((N, 2), np.float32)
    fade = int(0.008 * SR)
    for x in segs:
        if x["kind"] != "take": continue
        a, b = int(x["a"] * SR), int(x["b"] * SR); o = int(x["out"] * SR)
        chunk = sp[a:b].copy(); n = min(len(chunk), N - o)
        if n <= 0: continue
        chunk = chunk[:n]; ramp = np.linspace(0, 1, min(fade, n))[:, None]
        chunk[:len(ramp)] *= ramp; chunk[-len(ramp):] *= ramp[::-1]
        mix[o:o + n] += chunk
    for at, i in voices:
        v, _ = sf.read(voice_files[i], always_2d=True); o = int(at * SR); n = min(len(v), N - o)
        if n > 0: mix[o:o + n] += 0.8 * v[:n, :1].repeat(2, 1)
    mdir = os.path.join(HERE, "music")
    py = sys.executable
    if demo_seg:
        dcut = demo_seg["out"]
        cfile, dfile = os.path.join(work, "m_comedic.wav"), os.path.join(work, "m_demo.wav")
        subprocess.run([py, os.path.join(mdir, "comedic.py"), f"{dcut + 1}", f"{dcut}", f"{dcut}", f"{dcut}", f"{dcut + 0.5}", cfile], check=True, capture_output=True)
        subprocess.run([py, os.path.join(mdir, "demo.py"), f"{total - dcut}", f"{demo_seg['len']}", f"{end_at - dcut}", dfile], check=True, capture_output=True)
        c, _ = sf.read(cfile, always_2d=True); dm, _ = sf.read(dfile, always_2d=True)
        nc = min(len(c), int(dcut * SR)); mix[:nc] += 0.1 * c[:nc]
        t = np.arange(len(dm)) / SR
        back, blk, end = demo_seg["len"], black - dcut, end_at - dcut
        vol = np.interp(t, [0, back - 0.15, back + 0.25, blk - 0.1, blk, end - 0.05, end, t[-1] + 1], [0.6, 0.6, 0.13, 0.13, 0, 0, 0.55, 0.55])
        o = int(dcut * SR); n = min(len(dm), N - o); mix[o:o + n] += (dm[:n] * vol[:n, None])
        # whoosh into the cut
        wl = int(0.45 * SR); tw = np.arange(wl) / SR
        wh = np.random.default_rng(1).standard_normal(wl) * np.sin(np.pi * tw / tw[-1]) ** 2
        from pedalboard import Pedalboard, HighpassFilter, LowpassFilter
        wh = Pedalboard([HighpassFilter(500), LowpassFilter(6000)])(wh[None, :].astype(np.float32), SR)[0] * 0.25
        o = max(0, int((dcut - 0.4) * SR)); mix[o:o + wl, 0] += wh[:N - o]; mix[o:o + wl, 1] += wh[:N - o]
    else:
        cfile = os.path.join(work, "m_comedic.wav")
        subprocess.run([py, os.path.join(mdir, "comedic.py"), f"{total}", f"{total + 1}", f"{total + 1}", f"{total + 1}", f"{end_at}", cfile], check=True, capture_output=True)
        c, _ = sf.read(cfile, always_2d=True); n = min(len(c), N)
        t = np.arange(n) / SR
        mix[:n] += c[:n] * np.interp(t, [0, black - 0.1, black, end_at - 0.05, end_at, total + 1], [0.1, 0.1, 0, 0, 0.5, 0.5])[:, None]
    final = os.path.join(work, "final_yt.wav" if yt else "final.wav")
    A._finish(mix.T.copy(), SR, final)

    # 3) video
    grade = Grader(p.get("look", "daylight"), W, H)
    OW, OH = int(W * s), int(H * s)
    face = p.get("face") or dict(cx=0.5, cy=0.42, bottom=0.55)
    cap_top = int(np.clip(face["bottom"] * H + 70, 900, 1480) * s)
    fcap = font("Poppins-Bold.ttf", int(56 * s))
    cap_items = []
    for l in p.get("lines", []):
        if not l["text"].strip() or l.get("skip"): continue
        a, b = to_out(l["start"]), to_out(min(l["end"], l["start"] + 6))
        if a is None: continue
        if b is None:  # caption runs past a cut: end it at the cut
            seg = next(x for x in segs if x["kind"] == "take" and x["a"] - 1e-3 <= l["start"] <= x["b"] + 1e-3); b = seg["out"] + seg["b"] - seg["a"]
        sp_ = np.asarray(text_block(l["text"], fcap, 940 * s, stroke=int(5 * s), accent=(255, 210, 63, 255)))
        cap_items.append((a, b, sp_))
    hdr = np.asarray(header_box(p["header"], s)) if p.get("header") else None
    endimg = end_card(p, s) if has_end else None
    tmp = os.path.join(work, "video.mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium" if yt else "slow", "-crf", "16" if yt else "18", "-pix_fmt", "yuv420p", tmp],
                           stdin=subprocess.PIPE)
    nframes = int(total * FPS); fi = 0
    E.progress("render", 0.0, "Rendering")

    def emit_frame(x, t):
        nonlocal fi
        if fi % 15 == 0: E.progress("render", fi / nframes)
        enc.stdin.write(np.ascontiguousarray(x).tobytes()); fi += 1

    for x in segs:
        if x["kind"] == "demo":
            for k, fr in enumerate(demo.frames()):
                emit_frame(fr, x["out"] + k / FPS)
            continue
        n = round((x["b"] - x["a"]) * FPS)
        for k, fr in enumerate(E.frames(p["video"], x["a"], x["b"] + 0.1)):
            if k >= n: break
            t = x["out"] + k / FPS
            img = grade(fr, fi)
            z0, z1 = x["zoom"]; z = z0 + (z1 - z0) * k / max(1, n - 1)
            cxp, cyp = face["cx"] * W, face["cy"] * H
            M = np.float32([[z, 0, (1 - z) * cxp], [0, z, (1 - z) * cyp]])
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            if s != 1: img = cv2.resize(img, (OW, OH), interpolation=cv2.INTER_LANCZOS4)
            img = np.ascontiguousarray(img)
            if hdr is not None: blend(img, hdr, (OW - hdr.shape[1]) // 2, int(248 * s))
            for a, b, spr in cap_items:
                if a <= t < b:
                    al = min(1.0, (t - a) / 0.1, (b - t) / 0.1)
                    blend(img, spr, (OW - spr.shape[1]) // 2, cap_top, alpha=al)
            emit_frame(img, t)
        while fi < round((x["out"] + x["b"] - x["a"]) * FPS):  # pad if the source ran short
            emit_frame(img, 0)
    blackf = np.zeros((OH, OW, 3), np.uint8)
    while fi < round(end_at * FPS): emit_frame(blackf, 0)
    if endimg is not None:
        for k in range(round(4.0 * FPS)):
            a = min(1.0, k / 8); emit_frame((endimg.astype(np.float32) * a).astype(np.uint8), 0)
    while fi < nframes: emit_frame(blackf if endimg is None else endimg, 0)
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
