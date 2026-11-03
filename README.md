# Reel Studio

**Free, open-source reel editor that does the boring part.** Drop in a clip and it grades the colour, polishes the
sound, times lyrics or captions, and exports for Instagram and YouTube. Runs entirely on your computer. Bring your own
AI key (Claude or any OpenAI-compatible API) if you want it to fix mis-heard words, write your AI character's lines
and draft post captions. Without a key, the built-in rules do everything.

By [Laya Labs](https://layalabs.in).

## Templates
- **Singing cover**: separates voice and instrument, levels the vocal (phone mics lose you on some phrases), ballad
  reverb, steady instrument; lyrics timed to your phrases; song title card; mood looks.
- **Talking reel**: clean voice, bold captions.
- **AI skit**: you talk to an AI on your laptop and leave pauses. It answers in a natural offline voice (Kokoro) that
  sounds like it's coming from the laptop. Dead air is cut with alternating punch-ins; captions sit under your chin;
  optional **product demo** from a screen recording (beat-cut shots, punch-ins, light sweeps, text); comedic score +
  demo track; POV header; end card with your call to action.

Exports: Instagram (1080x1920, -14 LUFS) and YouTube Shorts 4K (higher-bitrate audio, mono-safe mix).

## Download
Mac (Apple Silicon) and Windows installers are on the [Releases](../../releases/latest) page. On first launch the app
downloads its AI models (about 1.2 GB) once.

The app isn't signed by Apple/Microsoft yet, so the first launch shows a warning. Mac: open it once, then System
Settings > Privacy & Security > **Open Anyway**. Windows: "Windows protected your PC" > **More info > Run anyway**.

## Is it safe?
Every release is built by GitHub Actions straight from this repository, never on a personal machine, and comes with:
- a signed **build-provenance attestation** (Sigstore). Verify any download with
  `gh attestation verify ReelStudio-mac-arm64.dmg --repo Prajwal434/reel-studio`;
- a **ClamAV virus scan** (`VIRUS-SCAN.txt` on the release);
- **SHA-256 checksums** (`SHA256SUMS.txt`).

## Build from source
- `engine/`: Python 3.12 engine (`engine.py <command> <project>`; JSON-line events on stdout).
- `desktop/`: Tauri 2 app (Rust shell + plain HTML/JS UI).
- `build/`: scripts that bundle a portable Python, the engine's packages and ffmpeg into the app.

```
# dev: run the UI against your own Python with the packages in engine/requirements.txt
cd desktop && npm install && npx tauri dev          # set the Python path in Settings
# release build with the engine inside
zsh build/bundle_engine_mac.sh          # or: pwsh build/bundle_engine_windows.ps1
cd desktop && npx tauri build
```

## Licence
MIT (see `LICENSE`). Fonts: Cormorant Garamond and Poppins, SIL Open Font License (`engine/fonts/`). Models downloaded
at first run keep their own licences (faster-whisper, BS-RoFormer via audio-separator, Kokoro, MediaPipe).
