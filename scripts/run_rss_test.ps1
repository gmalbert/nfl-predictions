# Powershell helper: regenerate RSS and run the link-resolution test locally
# Usage: Open PowerShell in repo root and run: .\scripts\run_rss_test.ps1

$ErrorActionPreference = 'Stop'

# Use venv python if present
$venvPython = Join-Path -Path $PSScriptRoot -ChildPath "..\venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = (Resolve-Path $venvPython).ProviderPath
} else {
    $python = "python"
}

Write-Host "Using Python: $python"

Write-Host "Installing minimal dependencies (pandas, requests)..."
& $python -m pip install --upgrade pip
& $python -m pip install pandas requests

Write-Host "Generating RSS feed..."
& $python scripts\generate_rss.py

Write-Host "Running RSS link test..."
& $python tests\test_rss_links.py

Write-Host "Done. Check data_files\alerts_feed.xml and test output."