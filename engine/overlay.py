"""Text on video. Each text block is rendered once to an RGBA sprite (PIL) and alpha-blended per frame with a fade.
Styles: lyric (serif, under the face), title (italic song name + small caps line), caption (bold sans, talking reels).
Coordinates are in 1080x1920 units; s scales them (s=2 for the 4K YouTube render)."""
import os, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def _font(name, size, weight=None):
    f = ImageFont.truetype(os.path.join(FONTS, name), size)
    if weight is not None:
        try: f.set_variation_by_axes([weight])
        except Exception: pass
    return f


def _wrap(text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if font.getlength(t) <= maxw or not cur: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines


def _sprite(rows, s, shadow=16, color=(244, 242, 238)):
    """rows: [(text, font, tracking_px, alpha)] stacked and centred -> (rgba float32, w, h)"""
    pad = int(40 * s)
    sizes = []
    for text, font, tr, _ in rows:
        w = sum(font.getlength(c) + tr for c in text) - tr if tr else font.getlength(text)
        a, d = font.getmetrics()
        sizes.append((int(w), a + d))
    W = max(w for w, _ in sizes) + 2 * pad
    H = sum(h for _, h in sizes) + 2 * pad
    txt = Image.new("L", (W, H), 0)
    al = Image.new("L", (W, H), 0)
    dt, da = ImageDraw.Draw(txt), ImageDraw.Draw(al)
    y = pad
    for (text, font, tr, a), (w, h) in zip(rows, sizes):
        x = (W - w) / 2
        if tr:
            for c in text:
                dt.text((x, y), c, font=font, fill=255); da.text((x, y), c, font=font, fill=int(255 * a)); x += font.getlength(c) + tr
        else:
            dt.text((x, y), text, font=font, fill=255); da.text((x, y), text, font=font, fill=int(255 * a))
        y += h
    m = np.asarray(txt, np.float32) / 255
    soft = np.asarray(txt.filter(ImageFilter.GaussianBlur(shadow * s / 2)), np.float32) / 255
    tight = np.asarray(txt.filter(ImageFilter.GaussianBlur(1.5 * s)), np.float32) / 255
    sh = np.clip(soft * 1.6 * 0.9 + tight * 0.9, 0, 0.92)
    a_txt = np.asarray(al, np.float32) / 255
    alpha = np.clip(a_txt + sh * (1 - m), 0, 1)
    col = np.float32(color) / 255
    rgb = np.where(m[..., None] > 0, col, 0).astype(np.float32) * (a_txt[..., None] / np.maximum(alpha[..., None], 1e-4))
    return np.dstack([rgb, alpha]).astype(np.float32)


class Item:
    def __init__(self, sprite, cx, top, start, end, fade, rise):
        self.sp, self.cx, self.top, self.start, self.end, self.fade, self.rise = sprite, cx, top, start, end, fade, rise

    def draw(self, frame, t):
        if t < self.start or t > self.end: return
        op = min((t - self.start) / self.fade, (self.end - t) / self.fade, 1.0)
        if op <= 0: return
        h, w = self.sp.shape[:2]
        x0 = int(self.cx - w / 2); y0 = int(self.top + (1 - op) * self.rise)
        H, W = frame.shape[:2]
        fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if fx1 <= fx0 or fy1 <= fy0: return
        sp = self.sp[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
        a = sp[..., 3:4] * op
        region = frame[fy0:fy1, fx0:fx1].astype(np.float32) / 255
        frame[fy0:fy1, fx0:fx1] = ((region * (1 - a) + sp[..., :3] * a) * 255 + 0.5).astype(np.uint8)


def build(project, s=1.0):
    """all overlay items for a project, times relative to the trimmed clip start"""
    t0 = project["trim"]["in"]
    items = []
    if project["template"] == "cover":
        f = _font("CormorantGaramond[wght].ttf", int(54 * s), 600)
        lines = [l for l in project["lines"] if l["text"].strip()]
        for i, l in enumerate(lines):
            nxt = lines[i + 1]["start"] if i + 1 < len(lines) else 1e9
            a, b = l["start"] - 0.15 - t0, min(l["end"] + 0.45, nxt - 0.05) - t0
            rows = [(r, f, 0, 1.0) for r in _wrap(l["text"], f, 900 * s)]
            sp = _sprite(rows, s)
            items.append(Item(sp, 540 * s, 1470 * s - 40 * s, a, b, 0.3, 8 * s))
        song = project.get("song") or {}
        if project.get("title_card", True) and song.get("title"):
            ft = _font("CormorantGaramond-Italic[wght].ttf", int(80 * s), 500)
            fs = _font("CormorantGaramond[wght].ttf", int(30 * s), 500)
            sub = (f"{song['artist']} · COVER" if song.get("artist") else "COVER").upper()
            sp = _sprite([(song["title"], ft, 0, 1.0), (sub, fs, int(6 * s), 0.75)], s, shadow=18)
            items.append(Item(sp, 540 * s, 300 * s - 40 * s, 0.4, 6.0, 0.8, 0))
    else:
        f = _font("Poppins-Bold.ttf", int(62 * s))
        for l in project["lines"]:
            if not l["text"].strip(): continue
            rows = [(r, f, 0, 1.0) for r in _wrap(l["text"], f, 860 * s)]
            sp = _sprite(rows, s, shadow=12, color=(255, 255, 255))
            items.append(Item(sp, 540 * s, project.get("caption_y", 1230) * s - 40 * s, l["start"] - t0, l["end"] - t0, 0.08, 10 * s))
    return items
