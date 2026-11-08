# Build the dsh-free standalone package.
#
#   powershell -ExecutionPolicy Bypass -File scripts\package-standalone.ps1
#
# Ships the bridge plus the WS protocol docs. No DSH plugin files.

param([string]$Version = "1.0.0")

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$stage = Join-Path $root "dist\dsh-vts-avatar-standalone-$Version"
$zip = Join-Path $root "dist\dsh-vts-avatar-standalone-$Version.zip"

if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
if (Test-Path $zip) { Remove-Item $zip -Force }
New-Item -ItemType Directory -Path $stage -Force | Out-Null

Copy-Item (Join-Path $root "src") (Join-Path $stage "src") -Recurse
Copy-Item (Join-Path $root "tests") (Join-Path $stage "tests") -Recurse
Copy-Item (Join-Path $root "scripts\install.ps1") (Join-Path $stage "scripts") -Force
Copy-Item (Join-Path $root "scripts\start.ps1") (Join-Path $stage "scripts") -Force
Copy-Item (Join-Path $root "config.example.toml") $stage
Copy-Item (Join-Path $root "pyproject.toml") $stage
Copy-Item (Join-Path $root "LICENSE") $stage

$docs = Join-Path $root "..\standalone"
Copy-Item (Join-Path $docs "README.md") (Join-Path $stage "README.md")
Copy-Item (Join-Path $docs "PROTOCOL.md") (Join-Path $stage "PROTOCOL.md")
New-Item -ItemType Directory -Path (Join-Path $stage "plugin") -Force | Out-Null
Copy-Item (Join-Path $docs "manifest.json") (Join-Path $stage "plugin\manifest.json")
Copy-Item (Join-Path $docs "example_client.py") (Join-Path $stage "plugin\example_client.py")

foreach ($junk in @("data", ".venv", "models", "dist", "__pycache__", ".pytest_cache")) {
    Get-ChildItem $stage -Recurse -Force -Directory -Filter $junk -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force
}

Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip -Force
Write-Host "built: $zip"
Write-Host "size:  $([math]::Round((Get-Item $zip).Length / 1MB, 2)) MB"
