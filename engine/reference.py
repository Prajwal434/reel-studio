"""Reference videos -> a style to start from: cut pace (average shot length), brightness, saturation, warmth -> the
nearest look, a transition, whether to cut on the beat."""
import numpy as np, cv2, subprocess


def measure(E, path):
    info = E.probe(path)
    dur = info["duration"]
    step = 1 / 6  # 6 fps is enough to see cuts and colour
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf", "fps=6,scale=96:170", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    prev, cuts, stats = None, 0, []
    while True:
        b = pr.stdout.read(96 * 170 * 3)
        if len(b) < 96 * 170 * 3: break
        f = np.frombuffer(b, np.uint8).reshape(170, 96, 3).astype(np.float32) / 255
        hsv = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255
        stats.append((f.mean(), hsv[..., 1].mean(), f[..., 0].mean() - f[..., 2].mean()))
        if prev is not None:
            h1 = cv2.calcHist([(f * 255).astype(np.uint8)], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
            h0 = cv2.calcHist([(prev * 255).astype(np.uint8)], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
            if cv2.compareHist(h0, h1, cv2.HISTCMP_BHATTACHARYYA) > 0.45: cuts += 1
        prev = f
    pr.wait()
    if not stats: return None
    st = np.array(stats)
    return dict(duration=dur, shot=dur / (cuts + 1), bright=float(st[:, 0].mean()), sat=float(st[:, 1].mean()), warm=float(st[:, 2].mean()))


def style(E, paths):
    ms = [m for m in (measure(E, p) for p in paths) if m]
    if not ms: return None
    avg = {k: float(np.median([m[k] for m in ms])) for k in ("shot", "bright", "sat", "warm")}  # median: one long take doesn't skew it
    if avg["sat"] < 0.06: look = "noir"
    elif avg["bright"] < 0.33 and avg["warm"] < 0: look = "cool_night"
    elif avg["bright"] < 0.36 or avg["sat"] < 0.2: look = "moody"
    elif avg["warm"] > 0.08: look = "warm_film"
    else: look = "clean"
    fast = avg["shot"] < 2.2
    return dict(shot=round(avg["shot"], 1), look=look, transition="whip" if fast else "crossfade", beat_sync=fast,
                music="upbeat" if fast else "none" if avg["shot"] > 6 else "upbeat",
                note=f"Matched your references: {'fast' if fast else 'relaxed'} pace (shots about {avg['shot']:.0f} s), "
                     f"{'dark' if avg['bright'] < 0.36 else 'bright'} and {'muted' if avg['sat'] < 0.25 else 'colourful'}: {look.replace('_', ' ')} look.")
