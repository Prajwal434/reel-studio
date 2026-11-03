# Build the self-contained engine for the Windows app (same as bundle_engine_mac.sh): portable CPython 3.12 + the
# engine's packages + ffmpeg.exe + the engine code. Output: desktop\src-tauri\engine-dist
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Out = Join-Path $Repo "desktop\src-tauri\engine-dist"
if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force -Path (Join-Path $Out "bin") | Out-Null
uv python install 3.12
$Py = (uv python find 3.12).Trim()
$PyHome = Split-Path -Parent $Py
Write-Host "python: $PyHome"
Copy-Item -Recurse $PyHome (Join-Path $Out "python")
foreach ($d in @("Lib\test", "Lib\idlelib", "Lib\tkinter", "Lib\turtledemo")) {
  $p = Join-Path $Out "python\$d"; if (Test-Path $p) { Remove-Item -Recurse -Force $p }
}
$OutPy = Join-Path $Out "python\python.exe"
uv pip install --python $OutPy --break-system-packages -r (Join-Path $Repo "engine\requirements.txt")
$Ff = (& $OutPy -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())").Trim()
Copy-Item $Ff (Join-Path $Out "bin\ffmpeg.exe")
robocopy (Join-Path $Repo "engine") (Join-Path $Out "app") /E /XD __pycache__ /XF *.pyc /NFL /NDL /NJH /NJS | Out-Null
& $OutPy -m compileall -q -j 0 (Join-Path $Out "python\Lib") (Join-Path $Out "app") | Out-Null
$size = (Get-ChildItem -Recurse $Out | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host ("engine-dist: {0:N2} GB" -f $size)
exit 0
