# MIC Paper Experiment - Step 2: Run 3-way comparison
# =============================================================
# Usage: Open a PowerShell terminal and run:
#   cd <repo_root>
#   .\paper\mic\run_experiment.ps1
#
# This runs 1890 optimization tasks (21 problems x 3 variants x 30 seeds)
# Estimated time: 12-15 minutes with 10 workers.
# Results are saved incrementally (every 1 min), so you can Ctrl+C and
# re-run this script to resume from where it left off.

$env:VAMOS_MIC_N_JOBS = "10"
$env:VAMOS_MIC_N_SEEDS = "30"
$env:VAMOS_MIC_N_EVALS = "50000"
$env:VAMOS_MIC_RESUME = "1"
$env:VAMOS_CHECKPOINT_INTERVAL_MIN = "1"

Set-Location $PSScriptRoot
python scripts\02_run_mic_experiment.py

Write-Host ""
Write-Host "=== Experiment complete ==="
Write-Host "Now generate paper assets with:"
Write-Host "  python scripts\\01_make_assets.py"
