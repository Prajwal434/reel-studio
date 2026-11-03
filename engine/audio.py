"""Audio rules.
cover: separated vocal + instrument -> vocal rider (phone mics lose the singer on some phrases), ballad vocal chain,
       a second leveling pass, reverb/echo tail that swells into pauses, steady polished instrument, -14 LUFS.
       yt=True keeps the instrument width mono-safe (phones fold to mono; a Haas delay would comb-filter).
talk:  clean speech: high-pass, gentle denoise, presence, compression, -14 LUFS."""
import numpy as np, soundfile as sf, pyloudnorm as pyln, noisereduce as nr
from pedalboard import (Pedalboard, HighpassFilter, LowShelfFilter, PeakFilter, HighShelfFilter, Compressor, Reverb,
                        Delay, Limiter, Gain, LowpassFilter)


def _level(x, sr, win=0.4):
    w = int(win * sr)
    return 10 * np.log10(np.convolve(x.astype(np.float64) ** 2, np.ones(w) / w, mode="same") + 1e-12)


def _smooth(g, sr, sec):
    k = max(3, int(sec * sr))
    h = np.hanning(k)
    return np.convolve(g, h / h.sum(), mode="same")


def ride(x, sr, target, lo=-6.0, hi=14.0):
    """pull each phrase toward a steady level; breaths (-62..-50 dB) get at most +4 dB (more raises separation artefacts)"""
    lvl = _level(x, sr)
    gain = np.where(lvl > -50, np.clip(target - lvl, lo, hi), np.where(lvl > -62, np.clip(target - lvl, 0, 4.0), 0))
    gain = _smooth(gain, sr, 0.3)
    return (x * 10 ** (gain / 20)).astype(np.float32), gain


def _finish(mix, sr, out):
    meter = pyln.Meter(sr)
    mix = Pedalboard([Gain(-14.0 - meter.integrated_loudness(mix.T)), Compressor(-16, 1.6, 20, 200), Limiter(-1.0, 100)])(mix.astype(np.float32), sr)
    mix *= 10 ** ((-14.0 - meter.integrated_loudness(mix.T)) / 20)
    mix /= max(1, np.abs(mix).max() / 0.89)
    sf.write(out, mix.T, sr, subtype="PCM_24")
    return meter.integrated_loudness(mix.T)


def mix_cover(vocals, inst, out, yt=False, inst_db=-3.0):
    v, sr = sf.read(vocals, always_2d=True)
    g, _ = sf.read(inst, always_2d=True)
    v = nr.reduce_noise(y=v.mean(axis=1).astype(np.float32), sr=sr, stationary=True, prop_decrease=0.5).astype(np.float32)
    v, rgain = ride(v, sr, -24.0)
    warm = Pedalboard([LowShelfFilter(220, 3.0)])(v[None, :], sr)[0]  # far-from-mic phrases also lost body
    w = np.clip(rgain / 8.0, 0, 1).astype(np.float32)
    v = v * (1 - w) + warm * w
    dry = Pedalboard([HighpassFilter(100), PeakFilter(280, -3.0, 1.0), PeakFilter(3200, 2.0, 0.8), PeakFilter(6800, -3.0, 3.0),
                      HighShelfFilter(11000, 2.5), Compressor(-26, 3.0, 10, 150), Compressor(-15, 2.5, 3, 70)])(v[None, :], sr)
    lvl = _level(dry[0], sr)
    med = float(np.median(lvl[lvl > -45])) if (lvl > -45).any() else -24.0
    dry = ride(dry[0], sr, med, lo=-4.0, hi=8.0)[0][None, :]
    verb_l = Pedalboard([Reverb(room_size=0.85, damping=0.6, wet_level=1.0, dry_level=0.0, width=1.0), HighpassFilter(300), LowpassFilter(7500)])(dry, sr)
    verb_r = Pedalboard([Delay(0.019, 0.0, 1.0), Reverb(room_size=0.86, damping=0.55, wet_level=1.0, dry_level=0.0, width=1.0), HighpassFilter(300), LowpassFilter(7500)])(dry, sr)
    dly = Pedalboard([HighpassFilter(600), LowpassFilter(5000), Delay(0.48, 0.3, 1.0)])(dry, sr)
    lvl = _level(dry[0], sr)
    gap = _smooth(np.clip((med - 4 - lvl) / 10, 0, 1), sr, 0.35).astype(np.float32)
    wet = 1 + 1.8 * gap  # the tail swells into pauses instead of leaving a hole
    voc = np.stack([dry[0] + 0.32 * wet * verb_l[0] + 0.07 * wet * dly[0],
                    dry[0] + 0.32 * wet * verb_r[0] + 0.056 * wet * dly[0]]).astype(np.float32)
    g = g.T.astype(np.float32)
    if g.shape[0] == 1: g = np.repeat(g, 2, 0)
    gtr = Pedalboard([HighpassFilter(75), PeakFilter(220, -2.0, 1.0), PeakFilter(2800, -2.0, 0.9), HighShelfFilter(9000, 1.5), Compressor(-22, 2.0, 15, 200)])(g, sr)
    gtr = gtr + 0.18 * Pedalboard([Reverb(room_size=0.6, damping=0.6, wet_level=1.0, dry_level=0.0, width=1.0), HighpassFilter(250)])(gtr, sr)
    if not yt:
        d = int(0.012 * sr); gtr[1] = np.concatenate([np.zeros(d, np.float32), gtr[1][:-d]])
    n = min(voc.shape[1], gtr.shape[1]); voc, gtr = voc[:, :n], gtr[:, :n]
    meter = pyln.Meter(sr)
    gl = meter.integrated_loudness(gtr.T)
    if np.isfinite(gl):
        gtr *= 10 ** ((meter.integrated_loudness(voc.T) + inst_db - gl) / 20)
    return _finish(voc + gtr, sr, out)


def clean_speech(src, out):
    x, sr = sf.read(src, always_2d=True)
    x = nr.reduce_noise(y=x.mean(axis=1).astype(np.float32), sr=sr, stationary=True, prop_decrease=0.6).astype(np.float32)
    y = Pedalboard([HighpassFilter(80), PeakFilter(300, -2.0, 1.0), PeakFilter(3000, 2.0, 0.9), PeakFilter(7000, -2.0, 3.0),
                    Compressor(-24, 3.0, 8, 120), Compressor(-14, 2.0, 3, 60)])(x[None, :], sr)
    return _finish(np.repeat(y, 2, 0), sr, out)
