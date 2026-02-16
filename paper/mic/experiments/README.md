# MIC Experiments

This folder stores auxiliary experiments for the MIC paper.

## 01 - Instance Selector

Script: `01_instance_selector.py`

Goal:
- Use the full instance set from the MIC runtime table
  (`UF + LSMOP + C-DTLZ + DC-DTLZ + MW`, 32 problems).
- Build a representative subset with family-stratified coverage.

Usage (from repo root):

```powershell
.\.venv\Scripts\python.exe paper\mic\experiments\01_instance_selector.py
```

Default behavior:
- Uses `--selection-pct 40.0` (40% of the full set) unless `--k-total` is provided.

Examples:

```powershell
.\.venv\Scripts\python.exe paper\mic\experiments\01_instance_selector.py --selection-pct 35
.\.venv\Scripts\python.exe paper\mic\experiments\01_instance_selector.py --k-total 12
.\.venv\Scripts\python.exe paper\mic\experiments\01_instance_selector.py --dry-run
```

Default outputs:
- `experiments/mic/instance_selection/representative_instances_mic_runtime_p<pct>_k<k>.csv`
- `experiments/mic/instance_selection/representative_instances_mic_runtime_p<pct>_k<k>.json`

