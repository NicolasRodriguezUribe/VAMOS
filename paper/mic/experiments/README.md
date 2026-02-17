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
python paper\mic\experiments\01_instance_selector.py
```

Default behavior:
- Uses `--selection-pct 30.0` (30% of the full set) unless `--k-total` is provided.

Examples:

```powershell
python paper\mic\experiments\01_instance_selector.py --selection-pct 35
python paper\mic\experiments\01_instance_selector.py --k-total 12
python paper\mic\experiments\01_instance_selector.py --dry-run
```

Default outputs:
- `experiments/mic/instance_selection/representative_instances_mic_runtime_p<pct>_k<k>.csv`
- `experiments/mic/instance_selection/representative_instances_mic_runtime_p<pct>_k<k>.json`

## 02 - SMAC3 Tuning on Representative Instances

Script: `02_smac3_tuning_representative_instances.py`

Goal:
- Tune NSGA-II with SMAC3 using the representative subset selected in Experiment 01.
- Export best configuration and top distinct configurations.

Usage (from repo root):

```powershell
python paper\mic\experiments\02_smac3_tuning_representative_instances.py
```

Default behavior:
- Reads the latest selection JSON from `experiments/mic/instance_selection/`.
- Uses `--budget 20000`, `--max-trials 120`, `--seeds 0,1,2`.
- Uses `--timeout-seconds 32400` (9h).
- Runs Optuna in distributed mode (`SQLite` study + process workers).
- Uses `--n-jobs -1` as CPU cores minus one.
- Exports `--top-k 5` distinct configurations.
- Enforces `--min-distinct 5` in strict mode by default (`--strict-min-distinct`).

Examples:

```powershell
python paper\mic\experiments\02_smac3_tuning_representative_instances.py --selection-file experiments\mic\instance_selection\representative_instances_mic_runtime_p30p0_k10.json
python paper\mic\experiments\02_smac3_tuning_representative_instances.py --instances cec2009_uf1,lsmop3,mw2 --max-trials 60 --top-k 5 --min-distinct 5
python paper\mic\experiments\02_smac3_tuning_representative_instances.py --timeout-seconds 0
python paper\mic\experiments\02_smac3_tuning_representative_instances.py --dry-run
```

Default outputs:
- `experiments/mic/smac3_tuning/<run>/selected_instances.json`
- `experiments/mic/smac3_tuning/<run>/tuning_history.json`
- `experiments/mic/smac3_tuning/<run>/tuning_history.csv`
- `experiments/mic/smac3_tuning/<run>/best_config_raw.json`
- `experiments/mic/smac3_tuning/<run>/best_config_active.json`
- `experiments/mic/smac3_tuning/<run>/best_config_resolved.json`
- `experiments/mic/smac3_tuning/<run>/top_configs_distinct.json`
- `experiments/mic/smac3_tuning/<run>/top_configs_distinct.csv`
- `experiments/mic/smac3_tuning/<run>/tuning_summary.json`

## 03 - Optuna Tuning on Representative Instances

Script: `03_optuna_tuning_representative_instances.py`

Goal:
- Tune NSGA-II with Optuna using the representative subset selected in Experiment 01.
- Export best configuration and top distinct configurations.

Usage (from repo root):

```powershell
python paper\mic\experiments\03_optuna_tuning_representative_instances.py
```

Default behavior:
- Reads the latest selection JSON from `experiments/mic/instance_selection/`.
- Uses `--budget 20000`, `--max-trials 120`, `--seeds 0,1,2`.
- Uses `--timeout-seconds 32400` (9h).
- Exports `--top-k 5` distinct configurations.
- Enforces `--min-distinct 5` in strict mode by default (`--strict-min-distinct`).

Examples:

```powershell
python paper\mic\experiments\03_optuna_tuning_representative_instances.py --selection-file experiments\mic\instance_selection\representative_instances_mic_runtime_p30p0_k10.json
python paper\mic\experiments\03_optuna_tuning_representative_instances.py --instances cec2009_uf1,lsmop3,mw2 --max-trials 60 --top-k 5 --min-distinct 5
python paper\mic\experiments\03_optuna_tuning_representative_instances.py --n-jobs -1
python paper\mic\experiments\03_optuna_tuning_representative_instances.py --timeout-seconds 0
python paper\mic\experiments\03_optuna_tuning_representative_instances.py --dry-run
```

Default outputs:
- `experiments/mic/optuna_tuning/<run>/selected_instances.json`
- `experiments/mic/optuna_tuning/<run>/tuning_history.json`
- `experiments/mic/optuna_tuning/<run>/tuning_history.csv`
- `experiments/mic/optuna_tuning/<run>/best_config_raw.json`
- `experiments/mic/optuna_tuning/<run>/best_config_active.json`
- `experiments/mic/optuna_tuning/<run>/best_config_resolved.json`
- `experiments/mic/optuna_tuning/<run>/top_configs_distinct.json`
- `experiments/mic/optuna_tuning/<run>/top_configs_distinct.csv`
- `experiments/mic/optuna_tuning/<run>/tuning_summary.json`
- `experiments/mic/optuna_tuning/<run>/optuna_study.sqlite3`
