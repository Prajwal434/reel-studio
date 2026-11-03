#!/bin/zsh
# Build the self-contained engine for the macOS app: portable CPython 3.12 (python-build-standalone via uv) + the
# engine's packages + a static ffmpeg + the engine code. Models are NOT bundled (downloaded on first run).
# Output: desktop/src-tauri/engine-dist  (Tauri copies it into Reel Studio.app/Contents/Resources/engine)
set -e
REPO=${0:A:h:h}
OUT=$REPO/desktop/src-tauri/engine-dist
rm -rf "$OUT"; mkdir -p "$OUT/bin"
uv python install 3.12 >/dev/null
PY=$(uv python find 3.12)
PYHOME=${PY:A:h:h}
echo "python: $PYHOME"
cp -R "$PYHOME" "$OUT/python"
rm -rf "$OUT/python/lib/python3.12/test" "$OUT/python/lib/python3.12/idlelib" "$OUT/python/lib/python3.12/tkinter" "$OUT/python/lib/python3.12/turtledemo"
uv pip install --python "$OUT/python/bin/python3" --break-system-packages -r "$REPO/engine/requirements.txt"
FF=$("$OUT/python/bin/python3" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
cp "$FF" "$OUT/bin/ffmpeg"; chmod +x "$OUT/bin/ffmpeg"
rsync -a --exclude __pycache__ --exclude '*.pyc' "$REPO/engine/" "$OUT/app/"
find "$OUT/python" -name __pycache__ -type d -prune -exec rm -rf {} +
# the installed app is read-only: compile once here so launches are fast
"$OUT/python/bin/python3" -m compileall -q -j 0 "$OUT/python/lib" "$OUT/app" >/dev/null 2>&1 || true
du -sh "$OUT"
