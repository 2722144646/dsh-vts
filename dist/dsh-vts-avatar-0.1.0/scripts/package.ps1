# Build a release zip for sharing.
#
#   powershell -ExecutionPolicy Bypass -File scripts\package.ps1
#
# Produces dist\dsh-vts-avatar-<version>.zip containing source, scripts,
# README, LICENSE and a clean config.example.toml. User data, models and the
# virtual environment are never included.

param(
    [string]$Version = "0.1.0"
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$dist = Join-Path $root "dist"
$stage = Join-Path $dist "dsh-vts-avatar-$Version"
$zip = Join-Path $dist "dsh-vts-avatar-$Version.zip"

if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
if (Test-Path $zip) { Remove-Item $zip -Force }
New-Item -ItemType Directory -Path $stage -Force | Out-Null

$items = @(
    "src", "scripts", "locale", "tests",
    "index.js", "cordis.patch.yml", "package.json", "pyproject.toml",
    "README.md", "LICENSE", "config.example.toml"
)
foreach ($item in $items) {
    $src = Join-Path $root $item
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $stage $item) -Recurse -Force
    }
}

# Never ship local state.
foreach ($junk in @("data", ".venv", "models", "dist", "__pycache__")) {
    $p = Join-Path $stage $junk
    if (Test-Path $p) { Remove-Item $p -Recurse -Force }
}

Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip -Force
Write-Host "built: $zip"
Write-Host "size:  $([math]::Round((Get-Item $zip).Length / 1MB, 2)) MB"
