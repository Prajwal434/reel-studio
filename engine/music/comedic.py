"""Original comedic score (intro/stings) + playful funk groove for skit-style explainer reels (humour + drama, subtle under the voice).
Sections are cue-driven: sneaky pizzicato intro -> hard stop with a hit on the punchline -> low tension drone ->
dramatic "DUN" stab -> playful F-dorian funk groove (slap bass, e-piano, claps) from the drop -> "ta-da" button on the end card.
Usage: music_fun.py <total_s> <stop_s> <dun_s> <drop_s> <end_s> <out.wav>"""
import sys, numpy as np, soundfile as sf, pyloudnorm as pyln
from pedalboard import Pedalboard, Reverb, LowpassFilter, HighpassFilter, Compressor

TOTAL, STOP, DUN, DROP, END = map(float, sys.argv[1:6]); OUT = sys.argv[6]
SR, BPM = 48000, 120
N = int(TOTAL * SR); beat = 60 / BPM; bar = 4 * beat
rng = np.random.default_rng(5)
hz = lambda m: 440 * 2 ** ((m - 69) / 12)
T0 = DROP - np.ceil(DROP / bar) * bar  # bar grid: a downbeat lands on the drop
buf = {k: np.zeros(N) for k in ("pizz", "marimba", "bass", "kick", "clap", "shaker", "snap", "brass", "timp", "drone")}


def put(name, t, x):
    s = int(t * SR)
    if s >= N or s + len(x) <= 0: return
    a = max(0, -s); s = max(0, s); e = min(N, s + len(x) - a)
    buf[name][s:e] += x[a:a + e - s]


def tt(d): return np.arange(int(d * SR)) / SR


def pizz(m, d=0.35):  # plucked string: bright attack, quick decay
    t = tt(d); f = hz(m)
    return sum(g * np.sin(2 * np.pi * f * k * t) for k, g in ((1, 1), (2, .5), (3, .25), (4, .12))) * np.exp(-t * 14) * np.minimum(1, t * 400)


def marimba(m, d=0.6):
    t = tt(d); f = hz(m)
    return (np.sin(2 * np.pi * f * t) + .35 * np.sin(2 * np.pi * f * 4 * t) * np.exp(-t * 30)) * np.exp(-t * 7) * np.minimum(1, t * 600)


def tuba(m, d):  # round staccato bass (oom-pah)
    t = tt(d); f = hz(m)
    x = sum(g * np.sin(2 * np.pi * f * k * t) for k, g in ((1, 1), (2, .45), (3, .2)))
    return x * np.minimum(1, t * 120) * np.exp(-t * 5)


def kick():
    t = tt(.3); return np.sin(2 * np.pi * (50 + 80 * np.exp(-t * 40)) * t) * np.exp(-t * 12)


def clap():
    t = tt(.2); x = np.zeros(len(t))
    for o in (0, .008, .017):  # three quick slaps = hand clap
        i = int(o * SR); x[i:] += rng.standard_normal(len(t) - i) * np.exp(-t[:len(t) - i] * 45)
    return x * .6


def noise(d, decay): t = tt(d); return rng.standard_normal(len(t)) * np.exp(-t * decay)


def brass(chord, d):  # orchestral stab: detuned saw stack, fast attack
    t = tt(d); x = np.zeros(len(t))
    for m in chord:
        for dt in (-.006, 0, .006):
            ph = (hz(m) * (1 + dt) * t) % 1
            x += 2 * ph - 1
    return x / (3 * len(chord)) * np.minimum(1, t * 80) * np.exp(-t * 2.2)


def timp(m, d=1.2):
    t = tt(d); return (np.sin(2 * np.pi * hz(m) * (1 + .15 * np.exp(-t * 12)) * t) * np.exp(-t * 3) + noise(d, 60) * .3)


# ---- 1. sneaky intro: staccato pizzicato tiptoe (F minor-ish, chromatic creep), snaps on 2 & 4
SNEAK = [65, None, 68, None, 70, 71, 72, None, 70, None, 68, None, 67, 68, 65, None]
for k in range(int((STOP + 1) / (beat / 2))):
    t = k * beat / 2
    if t >= STOP - 0.02: break
    m = SNEAK[k % 16]
    if m: put("pizz", t, pizz(m) * (.9 if k % 2 == 0 else .6))
    if k % 2 == 0: put("bass", t, tuba([41, 41, 44, 43][(k // 8) % 4] - (0 if k % 4 == 0 else -7), .28) * .8)
    if k % 4 == 2: put("snap", t, noise(.06, 90) * .5)

# ---- 2. the punchline hit: everything stops, one stab + timpani
put("brass", STOP, brass([53, 60, 65, 68], .5) * 1.2)
put("timp", STOP, timp(29, .9) * .8)
# ---- 3. tension: low tremolo drone creeping up to the DUN
dd = DUN - STOP - 0.15
if dd > 0.3:
    t = tt(dd)
    trem = .5 + .5 * np.sign(np.sin(2 * np.pi * 9 * t))
    dr = (np.sin(2 * np.pi * hz(36) * t) + .5 * np.sin(2 * np.pi * hz(37) * t * 1.002)) * trem * np.linspace(.15, .7, len(t))
    put("drone", STOP + .15, dr)
# ---- 4. the DUN: big minor stab + timpani roll into it
for j in range(6): put("timp", DUN - .3 + j * .05, timp(29, .3) * (.2 + j * .08))
put("brass", DUN, brass([41, 48, 53, 56, 60], .9) * 1.4)
put("timp", DUN, timp(29, 1.4))

# ---- 5. groove from the drop: playful funk in F dorian -- punchy kick/clap, 16th hats, slap bass, e-piano stabs,
# a cheeky synth call every 4 bars (cooler than cartoon, still light-hearted)
def slap(m, d=.22, pop=False):
    t = tt(d); f = hz(m)
    ph = (f * t) % 1
    x = (2 * ph - 1) * (0.6 if pop else 0.35) + np.sin(2 * np.pi * f * t)
    return x * np.minimum(1, t * 900) * np.exp(-t * (14 if pop else 9))


def epiano(chord, d=.32):
    t = tt(d); x = np.zeros(len(t))
    for m in chord:
        f = hz(m)
        x += np.sin(2 * np.pi * f * t) + .25 * np.sin(2 * np.pi * 2 * f * t) + .2 * np.sin(2 * np.pi * 7 * f * t) * np.exp(-t * 40)
    return x / len(chord) * np.minimum(1, t * 500) * np.exp(-t * 6)


def lead(m0, m1, d=.18):  # little pitch-glide pluck = the "wink"
    t = tt(d); f = hz(m0) + (hz(m1) - hz(m0)) * np.minimum(1, t / .06)
    ph = np.cumsum(f) / SR
    return (2 * (ph % 1) - 1) * np.exp(-t * 10) * np.minimum(1, t * 600)


for k in ("fbass", "keys", "lead", "ohat"): buf[k] = np.zeros(N)
PROG = [(41, [56, 60, 63, 67]), (41, [56, 60, 63, 67]), (46, [56, 60, 62, 65]), (46, [56, 60, 62, 65])]  # Fm9 Fm9 Bb9 Bb9
BASS = {0: (0, False), 3: (12, True), 6: (0, False), 8: (3, False), 10: (12, True), 11: (10, True), 14: (7, False)}
CALL = [(0, 72, 75), (2, 75, 77), (3, 77, 80), (6, 80, 79), (8, 75, 75)]  # (16th, from, to)
nb = int(np.ceil((END - DROP) / bar))
for b in range(nb):
    t0 = DROP + b * bar
    root, chord = PROG[b % 4]
    for k in range(16):
        tk = t0 + k * beat / 4
        if tk >= END: break
        if k in BASS:
            o, pop = BASS[k]; put("fbass", tk, slap(root + o, .2 if pop else .3, pop) * (.8 if pop else 1))
        if k in (0, 7, 10) or (k == 15 and b % 2): put("kick", tk, kick())
        if k in (4, 12): put("clap", tk, clap())
        put("shaker", tk, noise(.04, 90) * (.4 if k % 4 == 2 else .16))
        if k == 14: put("ohat", tk, noise(.25, 10) * .25)
        if k in (6, 14) or (k == 3 and b % 2): put("keys", tk, epiano(chord))
    if b % 4 == 1 or b % 4 == 3:
        for k, m0, m1 in CALL:
            tk = t0 + k * beat / 4
            if tk < END: put("lead", tk, lead(m0, m1) * (.8 if b % 4 == 1 else .65))

# ---- 6. ta-da button on the end card (F major: the "aha" resolution)
put("brass", END, brass([53, 57, 60, 65], .25) * .9)
put("brass", END + .22, brass([53, 57, 60, 65, 69], 1.6) * 1.1)
put("timp", END + .22, timp(29, 1.2) * .7)
put("keys", END + .22, epiano([65, 69, 72, 77], 1.2) * .8)

fx = lambda name, board: board(buf[name].astype(np.float32), SR)
pn = lambda x: x / (np.abs(x).max() + 1e-9)
mix = (.55 * pn(fx("pizz", Pedalboard([HighpassFilter(180), Reverb(room_size=.35, wet_level=.18)])))
       + .62 * pn(fx("fbass", Pedalboard([LowpassFilter(2200)]))) + .45 * pn(fx("keys", Pedalboard([Reverb(room_size=.35, wet_level=.2)])))
       + .28 * pn(fx("lead", Pedalboard([LowpassFilter(4000), Reverb(room_size=.4, wet_level=.25)]))) + .12 * pn(fx("ohat", Pedalboard([HighpassFilter(7000)])))
       + .6 * pn(fx("bass", Pedalboard([LowpassFilter(900)])))
       + .6 * pn(buf["kick"]) + .32 * pn(fx("clap", Pedalboard([HighpassFilter(600), Reverb(room_size=.3, wet_level=.25)])))
       + .12 * pn(fx("shaker", Pedalboard([HighpassFilter(6000)]))) + .25 * pn(fx("snap", Pedalboard([HighpassFilter(1500)])))
       + .7 * pn(fx("brass", Pedalboard([LowpassFilter(3500), Reverb(room_size=.7, wet_level=.35)])))
       + .55 * pn(fx("timp", Pedalboard([LowpassFilter(2000), Reverb(room_size=.6, wet_level=.3)])))
       + .3 * pn(fx("drone", Pedalboard([LowpassFilter(400)]))))
t = np.arange(N) / SR
# intro sits a bit lower than the groove (keeps the drop meaningful)
mix *= np.where(t < DROP, .75, 1.0)
L, R = mix, np.concatenate([np.zeros(int(.009 * SR)), mix[:-int(.009 * SR)]])
st = Pedalboard([Compressor(-16, 2.5, 5, 120), HighpassFilter(30)])(np.stack([L, R]).astype(np.float32), SR).T
fo = int(.6 * SR); st[-fo:] *= np.linspace(1, 0, fo)[:, None]
st = pyln.normalize.loudness(st, pyln.Meter(SR).integrated_loudness(st), -14.0)
st /= max(1, np.abs(st).max() / .89)
sf.write(OUT, st, SR)
print("wrote", OUT, f"{TOTAL:.1f}s: stop {STOP:.2f} dun {DUN:.2f} drop {DROP:.2f} end {END:.2f}")
