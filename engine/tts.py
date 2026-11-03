"""AI voice for skit reels: Kokoro (offline neural TTS, Mac + Windows) -> 'playing through the laptop' chain
(small-speaker band, a bit of room) so it sounds like the AI is answering from the computer on camera."""
import os, urllib.request, numpy as np, soundfile as sf, pyloudnorm as pyln
from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, PeakFilter, Reverb, Compressor

VOICES = {  # id: label (male voices first: the default AI character)
    "am_michael": "Michael (US, calm)", "am_adam": "Adam (US)", "am_eric": "Eric (US)", "am_onyx": "Onyx (US, deep)",
    "bm_george": "George (UK)", "bm_lewis": "Lewis (UK)", "af_heart": "Heart (US, female)", "af_bella": "Bella (US, female)",
    "bf_emma": "Emma (UK, female)", "hm_omega": "Omega (Hindi accent)",
}
URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
FILES = ["kokoro-v1.0.onnx", "voices-v1.0.bin"]
_k = None


def ensure_model(models_dir, progress=None):
    d = os.path.join(models_dir, "kokoro"); os.makedirs(d, exist_ok=True)
    for f in FILES:
        p = os.path.join(d, f)
        if os.path.exists(p): continue
        tmp = p + ".part"
        with urllib.request.urlopen(URL + f, timeout=60) as r, open(tmp, "wb") as out:
            total = int(r.headers.get("Content-Length") or 0); got = 0
            while True:
                b = r.read(1 << 20)
                if not b: break
                out.write(b); got += len(b)
                if progress and total: progress(got / total)
        os.replace(tmp, p)
    return d


def synth(text, out, models_dir, voice="am_michael", speed=1.0, laptop=True):
    """-> seconds of speech written to `out` (48 kHz mono, -16 LUFS)"""
    global _k
    from kokoro_onnx import Kokoro
    d = ensure_model(models_dir)
    if _k is None: _k = Kokoro(os.path.join(d, FILES[0]), os.path.join(d, FILES[1]))
    a, sr = _k.create(text, voice=voice, speed=speed, lang="en-gb" if voice.startswith("b") else "en-us")
    a = np.asarray(a, np.float32)
    import librosa
    a = librosa.resample(a, orig_sr=sr, target_sr=48000); sr = 48000
    speech = len(a) / sr
    a = np.concatenate([a, np.zeros(int(0.5 * sr), np.float32)])
    if laptop:
        a = Pedalboard([HighpassFilter(300), LowpassFilter(8000), PeakFilter(900, 2.5, 1.2), PeakFilter(2500, 2.0, 1.0),
                        Compressor(-20, 3.0, 5, 80), Reverb(room_size=0.25, damping=0.6, wet_level=0.12, dry_level=0.9, width=0.0)])(a[None, :], sr)[0]
    a = pyln.normalize.loudness(a, pyln.Meter(sr).integrated_loudness(a), -16.0)
    sf.write(out, a / max(1, np.abs(a).max() / 0.95), sr)
    return speech
