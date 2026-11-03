"""Colour looks, applied evenly over the whole frame (no subject mask: mask edges read as a halo around the person).
Each look is a set of numbers for one grade() function: exposure, highlight roll-off, saturation, S-curve, matte
blacks, split-tone, halation, top fall-off, vignette, grain."""
import numpy as np, cv2

LOOKS = {
    "moody": dict(label="Moody", desc="Muted, teal shadows, warm skin, faded blacks, grain. Emotional songs.",
                  exp=0.86, knee=0.5, roll=0.45, sat=0.72, curve=0.22, matte=0.045, sh=(0.9, 1.0, 1.1), hi=(1.06, 1.0, 0.9),
                  hal=(0.25, 0.14, 0.08), top=0.4, vig=0.42, grain=0.018),
    "warm_film": dict(label="Warm film", desc="Golden, soft contrast, light grain. Love songs, daytime.",
                      exp=0.95, knee=0.6, roll=0.6, sat=0.88, curve=0.18, matte=0.03, sh=(1.0, 0.98, 0.94), hi=(1.07, 1.0, 0.88),
                      hal=(0.3, 0.18, 0.08), top=0.15, vig=0.3, grain=0.014),
    "clean": dict(label="Clean", desc="Natural colour, a little contrast. Talking / explainer reels.",
                  exp=1.0, knee=0.78, roll=0.7, sat=1.04, curve=0.2, matte=0.0, sh=(0.98, 1.0, 1.02), hi=(1.02, 1.0, 0.98),
                  hal=(0, 0, 0), top=0.0, vig=0.18, grain=0.0),
    "cool_night": dict(label="Cool night", desc="Blue, dim, quiet. Late-night, sad songs.",
                       exp=0.88, knee=0.55, roll=0.5, sat=0.7, curve=0.2, matte=0.04, sh=(0.86, 0.97, 1.14), hi=(1.0, 1.0, 1.03),
                       hal=(0.12, 0.14, 0.18), top=0.3, vig=0.42, grain=0.018),
    "daylight": dict(label="Daylight fix", desc="Phone take by a bright window: recovers the wall, adds contrast and depth.",
                     exp=0.94, knee=0.6, roll=0.42, sat=0.92, curve=0.26, matte=0.015, sh=(0.97, 1.0, 1.03), hi=(1.01, 1.0, 0.97),
                     hal=(0, 0, 0), top=0.32, vig=0.34, grain=0.006),
    "noir": dict(label="Black & white", desc="Classic B&W with deep blacks and grain.",
                 exp=0.95, knee=0.65, roll=0.55, sat=0.0, curve=0.3, matte=0.03, sh=(1, 1, 1), hi=(1, 1, 1),
                 hal=(0.12, 0.12, 0.12), top=0.25, vig=0.38, grain=0.024),
}


class Grader:
    def __init__(self, look, w, h, seed=3):
        self.L = LOOKS[look]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        self.top = (1 - self.L["top"] * np.clip((h * 0.47 - yy) / (h * 0.47), 0, 1))[..., None]
        self.vig = (1 - self.L["vig"] * np.clip(((xx - w * 0.5) / (w * 0.68)) ** 2 + ((yy - h * 0.42) / (h * 0.58)) ** 2, 0, 1) ** 1.3)[..., None]
        rng = np.random.default_rng(seed)
        self.grain = [cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 0.8) for _ in range(8)] if self.L["grain"] else None
        self.hal = np.float32(self.L["hal"])
        self.sh, self.hi = np.float32(self.L["sh"]), np.float32(self.L["hi"])

    def __call__(self, rgb, i=0):
        L = self.L
        x = rgb.astype(np.float32) / 255 * L["exp"]
        k = L["knee"]
        x = np.where(x > k, k + (x - k) * L["roll"], x)  # highlight roll-off: bright walls stop competing with the face
        g = x.mean(axis=2, keepdims=True)
        x = np.clip(g + (x - g) * L["sat"], 0, 1)
        x = x + L["curve"] * (x - 0.5) * (1 - np.abs(2 * x - 1))
        x = L["matte"] + x * (1 - 1.5 * L["matte"])
        lum = x.mean(axis=2, keepdims=True)
        x = x * (self.sh * (1 - lum) + self.hi * lum)
        if self.hal.any():
            hl = np.clip(x.mean(axis=2) - 0.5, 0, 1)
            x = x + cv2.GaussianBlur(hl, (0, 0), 22)[..., None] * self.hal
        x = x * self.top * self.vig
        if self.grain is not None:
            x = x + self.grain[i % 8][..., None] * L["grain"] * (0.6 + 0.4 * (1 - lum))
        return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
