# MIC 2026 regular paper (AOS-only)

This folder contains an LNCS-formatted draft for a **regular paper** submission focused on **Adaptive Operator Selection (AOS)** in NSGA-II.

## Quickstart

From the repo root:

1) Generate tables/figures from the committed CSV artifacts:

```powershell
.\.venv\Scripts\python.exe paper_MIC\scripts\01_make_assets.py
```

2) Compile the LNCS paper:

```powershell
cd paper_MIC
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Outputs:
- `paper_MIC/main.pdf`

## Notes

- The asset script reads:
  - `experiments/ablation_aos_racing_tuner.csv` (filters to `baseline` vs `aos`)
  - `experiments/ablation_aos_anytime.csv` (optional convergence plot)
  - `experiments/ablation_aos_trace.csv` (optional operator-usage plot)
- Update author list, affiliations, and final wording in `paper_MIC/main.tex`.

