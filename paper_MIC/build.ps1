Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
try {
    # Regenerate tables/figures
    & ..\.venv\Scripts\python.exe .\scripts\01_make_assets.py

    # Build PDF
    latexmk -pdf -interaction=nonstopmode -halt-on-error .\main.tex
} finally {
    Pop-Location
}

