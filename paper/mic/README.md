# MIC 2026 regular paper (AOS-only)

This folder contains an LNCS-formatted draft for a **regular paper** submission focused on **Adaptive Operator Selection (AOS)** in NSGA-II.

## Quickstart

From the repo root:

1) Generate tables/figures from the committed CSV artifacts:

```powershell
.\.venv\Scripts\python.exe paper\mic\scripts\01_make_assets.py
```

2) Compile the LNCS paper:

```powershell
cd paper\mic
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Outputs:
- `paper/mic/main.pdf`

## Notes

- The asset script reads:
  - `experiments/ablation_aos_racing_tuner.csv` (filters to `baseline` vs `aos`)
  - `experiments/ablation_aos_anytime.csv` (optional convergence plot)
  - `experiments/ablation_aos_trace.csv` (optional operator-usage plot)
- Update author list, affiliations, and final wording in `paper/mic/main.tex`.

## 2x2 factorial experiment (Archive x AOS)

To isolate causal effects, run:

```powershell
$env:VAMOS_MIC_VARIANTS="baseline,aos,baseline_archive,aos_archive"
$env:VAMOS_MIC_OUTPUT_CSV="experiments/mic/mic_factorial_archive_aos.csv"
$env:VAMOS_MIC_ANYTIME_CSV="experiments/mic/mic_factorial_archive_aos_anytime.csv"
$env:VAMOS_MIC_TRACE_CSV="experiments/mic/mic_factorial_archive_aos_trace.csv"
$env:VAMOS_MIC_TRACE_VARIANTS="aos,aos_archive"
# Optional archive controls:
# $env:VAMOS_MIC_ARCHIVE_SIZE="100"
# $env:VAMOS_MIC_ARCHIVE_TYPE="hypervolume"   # or "crowding"
# $env:VAMOS_MIC_ARCHIVE_UNBOUNDED="0"        # set "1" for unbounded archive
.\.venv\Scripts\python.exe paper\mic\scripts\02_run_mic_experiment.py
```
