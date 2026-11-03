"""Product-demo track (original): starts on the reveal smash cut.
[0, R)  the drop: 125 BPM, four-on-the-floor kick, claps, open hats, sidechained supersaw chords (Am-F-C-G), 16th pluck
        arpeggio, sub bass. A tight downlifter + impact on the first downbeat.
[R, E)  back in the skit: same progression, filtered right down, no drums (sits under dialogue).
E       end card: big Am(add9) chord hit + sub drop, long tail.
Usage: music_demo.py <total_s> <reveal_s> <endcard_s> <out.wav>   (times relative to the reveal start)"""
import sys, numpy as np, soundfile as sf, pyloudnorm as pyln
from pedalboard import Pedalboard, Reverb, LowpassFilter, HighpassFilter, Compressor, Delay

TOTAL, R, E = map(float, sys.argv[1:4]); OUT = sys.argv[4]
SR, BPM = 48000, 125
N = int(TOTAL * SR); beat = 60 / BPM; bar = 4 * beat
rng = np.random.default_rng(11)
hz = lambda m: 440 * 2 ** ((m - 69) / 12)
tt = lambda d: np.arange(int(d * SR)) / SR
bus = {k: np.zeros(N, np.float32) for k in ("kick", "clap", "hat", "chords", "pluck", "sub", "fx", "pad", "end")}


def put(name, t, x):
    s = int(t * SR)
    if s >= N: return
    e = min(N, s + len(x)); bus[name][s:e] += x[:e - s].astype(np.float32)


def saw(f, t, detune=(-0.11, -0.04, 0, 0.04, 0.11)):  # supersaw: detuned stack (cents/100 as ratio offsets)
    return sum(2 * ((f * (1 + d / 100) * t + rng.random()) % 1) - 1 for d in detune) / len(detune)


def kick():
    t = tt(0.35); return np.sin(2 * np.pi * (45 + 110 * np.exp(-t * 32)) * t) * np.exp(-t * 9) + 0.2 * rng.standard_normal(len(t)) * np.exp(-t * 200)


def clap():
    t = tt(0.25); x = np.zeros(len(t))
    for o in (0, .009, .019):
        i = int(o * SR); x[i:] += rng.standard_normal(len(t) - i) * np.exp(-t[:len(t) - i] * 38)
    return x * 0.5


def noise(d, decay): t = tt(d); return rng.standard_normal(len(t)) * np.exp(-t * decay)


PROG = [(57, [57, 60, 64, 69]), (53, [57, 60, 65, 69]), (48, [55, 60, 64, 67]), (55, [55, 59, 62, 67])]  # Am F C G
ARP = [0, 2, 1, 3, 2, 1, 3, 2]

# ---- the drop
put("fx", 0, noise(0.6, 7) * 0.5)                      # impact noise burst on the cut
t = tt(0.9); put("fx", 0, np.sin(2 * np.pi * (60 * np.exp(-t * 2.5)) * t) * np.exp(-t * 3) * 0.9)  # sub boom
nb = int(np.ceil(R / bar))
for b in range(nb):
    root, chord = PROG[b % 4]
    t0 = b * bar
    for k in range(16):
        tk = t0 + k * beat / 4
        if tk >= R: break
        if k % 4 == 0: put("kick", tk, kick())
        if k in (4, 12): put("clap", tk, clap())
        if k % 4 == 2: put("hat", tk, noise(0.18, 22) * 0.5)
        else: put("hat", tk, noise(0.03, 120) * 0.25)
        m = chord[ARP[k % 8]] + 12
        tp = tt(0.16); put("pluck", tk, saw(hz(m), tp, (0, 0.06)) * np.exp(-tp * 22))
    td = tt(min(bar, R - t0))
    put("chords", t0, sum(saw(hz(n), td) for n in chord) / 4)
    put("sub", t0, np.sin(2 * np.pi * hz(root - 24) * td))
# riser into the end of the drop is not needed: the cut back to the skit is a hard stop
# ---- under the punchline: filtered progression, no drums
for b in range(int(np.ceil((E - R) / bar))):
    root, chord = PROG[b % 4]; t0 = R + b * bar
    td = tt(min(bar, E - t0))
    if len(td): put("pad", t0, sum(np.sin(2 * np.pi * hz(n) * td) + 0.3 * saw(hz(n), td, (0, 0.05)) for n in chord) / 4)
# ---- end card hit
t = tt(min(4.5, TOTAL - E))
if len(t):
    put("end", E, sum(saw(hz(n), t) for n in (57, 64, 69, 71, 72, 76)) / 6 * np.exp(-t * 0.9))
    put("sub", E, np.sin(2 * np.pi * hz(33) * t) * np.exp(-t * 1.2))
    put("kick", E, kick())

# sidechain: chords/sub/pluck duck on every kick in the drop
sc = np.ones(N, np.float32)
for b in range(int(R / beat) + 1):
    s = int(b * beat * SR); n = int(beat * SR)
    env = 1 - 0.7 * np.exp(-np.arange(n) / SR * 9)
    sc[s:s + n] = env[:max(0, min(n, N - s))]
for k in ("chords", "sub", "pluck"): bus[k] *= sc

fx = lambda k, board: board(bus[k][None, :], SR)[0]
pn = lambda x: x / (np.abs(x).max() + 1e-9)
mix = (0.95 * pn(bus["kick"]) + 0.42 * pn(fx("clap", Pedalboard([HighpassFilter(500), Reverb(room_size=0.35, wet_level=0.25)])))
       + 0.22 * pn(fx("hat", Pedalboard([HighpassFilter(7000)])))
       + 0.5 * pn(fx("chords", Pedalboard([HighpassFilter(180), LowpassFilter(6500), Reverb(room_size=0.6, wet_level=0.25)])))
       + 0.32 * pn(fx("pluck", Pedalboard([HighpassFilter(400), Delay(beat * 0.75, 0.3, 0.25), Reverb(room_size=0.4, wet_level=0.2)])))
       + 0.6 * pn(fx("sub", Pedalboard([LowpassFilter(160)])))
       + 0.55 * pn(fx("fx", Pedalboard([LowpassFilter(5000), Reverb(room_size=0.7, wet_level=0.3)])))
       + 0.28 * pn(fx("pad", Pedalboard([LowpassFilter(900), Reverb(room_size=0.8, wet_level=0.4)])))
       + 0.6 * pn(fx("end", Pedalboard([LowpassFilter(7000), Reverb(room_size=0.9, wet_level=0.45)]))))
L, Rr = mix, np.concatenate([np.zeros(int(0.011 * SR), np.float32), mix[:-int(0.011 * SR)]])
st = Pedalboard([Compressor(-14, 2.5, 5, 100), HighpassFilter(28)])(np.stack([L, Rr]).astype(np.float32), SR).T
fo = int(0.8 * SR); st[-fo:] *= np.linspace(1, 0, fo)[:, None]
st = pyln.normalize.loudness(st, pyln.Meter(SR).integrated_loudness(st), -14.0)
st /= max(1, np.abs(st).max() / 0.89)
sf.write(OUT, st, SR)
print("wrote", OUT, f"{TOTAL:.2f}s  drop 0-{R:.2f}  under-dialogue {R:.2f}-{E:.2f}  end {E:.2f}")
