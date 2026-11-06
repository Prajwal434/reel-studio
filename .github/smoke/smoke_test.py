"""End-to-end check of the packaged engine, run on a real Windows (or Mac) machine by .github/workflows/windows-smoke.yml.
Does what a first-time user does: download the models, then make a talking reel, a clips reel and a singing cover.
Usage: <bundled python> smoke_test.py <engine.py> <work dir>"""
import json, os, subprocess, sys, time

ENGINE, WORK = sys.argv[1], sys.argv[2]
if os.environ.get("RS_BIN"):  # the bundled ffmpeg, the same way the engine finds it
    os.environ["PATH"] = os.environ["RS_BIN"] + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, os.path.dirname(ENGINE))
os.makedirs(WORK, exist_ok=True)
FF = "ffmpeg"
results = []


def engine(*args, timeout=1800):
    t = time.time()
    p = subprocess.run([sys.executable, ENGINE, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    events = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    last = next((e for e in reversed(events) if e.get("type") in ("done", "error")), None)
    if not last or last["type"] == "error" or p.returncode != 0:
        raise RuntimeError(f"engine {args[0]} failed (exit {p.returncode}): {last and last.get('msg')}\n{p.stderr[-3000:]}")
    print(f"   engine {args[0]}: ok in {time.time() - t:.0f}s", flush=True)
    return last


def duration(path):
    import av
    with av.open(path) as c: return c.duration / 1e6


def step(name, fn):
    print(f"== {name}", flush=True)
    try:
        fn(); results.append((name, True, "")); print(f"PASS {name}", flush=True)
    except Exception as e:
        results.append((name, False, str(e))); print(f"FAIL {name}: {e}", flush=True)


def project(name, data):
    d = os.path.join(WORK, name); os.makedirs(d, exist_ok=True)
    json.dump(data, open(os.path.join(d, "project.json"), "w"))
    return d


def first_run():
    engine("setup", timeout=3600)
    r = engine("check")
    if not r.get("ok"): raise RuntimeError(f"engine check not ok: {json.dumps(r)}")


SPEECH = os.path.join(WORK, "speech.wav")
TALK = os.path.join(WORK, "talk.mp4")


def make_media():
    import tts
    tts.synth("Hello! This is a quick test of Reel Studio on Windows. If you can read these words as captions, the editor works.",
              SPEECH, os.environ["RS_MODELS"], laptop=False)
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=30", "-i", SPEECH, "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", TALK], check=True)
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=30", "-t", "4", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", os.path.join(WORK, "landscape.mp4")], check=True)
    # a "song": the speech over a soft chord, for the singing-cover template (voice / instrument separation)
    subprocess.run([FF, "-v", "error", "-y", "-i", SPEECH, "-f", "lavfi", "-i", "sine=f=220:d=12", "-f", "lavfi", "-i", "sine=f=330:d=12",
                    "-filter_complex", "[1][2]amix=2,volume=0.3[m];[0][m]amix=2:duration=first[a]", "-map", "[a]", "-ac", "2",
                    os.path.join(WORK, "song.wav")], check=True)
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=30", "-i", os.path.join(WORK, "song.wav"),
                    "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", os.path.join(WORK, "cover.mp4")], check=True)


def talk_reel():
    d = project("talk", {"name": "windows talk", "template": "talk", "video": TALK})
    p = engine("analyze", d)["project"]
    words = " ".join(l["text"] for l in p.get("lines", [])).lower()
    if "reel" not in words and "windows" not in words and "test" not in words: raise RuntimeError(f"captions look wrong: {words[:200]!r}")
    out = engine("render", d, "ig")["path"]
    if duration(out) < 3: raise RuntimeError("export too short")


def clips_reel():
    d = project("clips", {"name": "windows clips", "template": "montage", "clips": [{"path": TALK}, {"path": os.path.join(WORK, "landscape.mp4")}],
                          "options": {"transition": "whip", "music": "upbeat"}})
    engine("analyze", d)
    out = engine("render", d, "ig")["path"]
    if duration(out) < 3: raise RuntimeError("export too short")


def cover():
    d = project("cover", {"name": "windows cover", "template": "cover", "video": os.path.join(WORK, "cover.mp4"),
                          "song": {"title": "Test Song", "artist": "Reel Studio"}})
    engine("analyze", d, timeout=3600)
    out = engine("render", d, "ig")["path"]
    if duration(out) < 3: raise RuntimeError("export too short")


step("first launch: download models + self-check", first_run)
step("make test media (AI voice + ffmpeg)", make_media)
step("talking reel: analyze + export", talk_reel)
step("clips + transitions: analyze + export", clips_reel)
step("singing cover: separate voice/instrument + export", cover)
print("\n" + "\n".join(f"{'PASS' if ok else 'FAIL'}  {n}" + (f"\n      {e[:400]}" if e else "") for n, ok, e in results))
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
