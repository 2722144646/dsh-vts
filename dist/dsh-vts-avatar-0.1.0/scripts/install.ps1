# One-time setup for dsh-vts-avatar.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1
#
# Creates .venv, installs the package, and checks the voice model.

param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$venv = Join-Path $root ".venv"
$pythonExe = Join-Path $venv "Scripts\python.exe"

Write-Host "== dsh-vts-avatar setup =="
Write-Host "project: $root"

$config = Join-Path $root "config.toml"
$example = Join-Path $root "config.example.toml"
if (-not (Test-Path $config) -and (Test-Path $example)) {
    Copy-Item $example $config
    Write-Host "created config.toml from config.example.toml"
}

if (-not (Test-Path $pythonExe)) {
    Write-Host "creating virtual environment..."
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        & uv venv --python 3.13 $venv
    } else {
        & $Python -m venv $venv
    }
}

Write-Host "installing package..."
& $pythonExe -m pip install --upgrade pip | Out-Null
& $pythonExe -m pip install -e $root

$voice = Join-Path $root "models\feiyu_distill_v8_final.onnx"
if (-not (Test-Path $voice)) {
    Write-Host ""
    Write-Warning "Voice model missing: models\feiyu_distill_v8_final.onnx"
    Write-Host "Put a Piper Chinese voice (.onnx + .onnx.json) into models\,"
    Write-Host "then set voice_model / voice_config in config.toml."
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Open VTube Studio, allow the API plugin."
Write-Host "  2. Check config.toml (voice paths, ffmpeg_path if postprocess is on)."
Write-Host "  3. Start the bridge:  powershell -File scripts\start.ps1"
Write-Host "  4. Enable the dsh-vts-avatar plugin in dsh."
