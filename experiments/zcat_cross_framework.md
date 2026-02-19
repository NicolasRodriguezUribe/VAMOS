# ZCAT Cross-Framework Experiment

This experiment compares `VAMOS`, `jMetalPy`, and `pymoo` on `ZCAT1..20` using a shared protocol.

## Manifests

- Full study: `experiments/configs/zcat_cross_framework.yaml`
- Smoke study: `experiments/configs/zcat_cross_framework_smoke.yaml`

## Commands

Run from repo root:

```powershell
python experiments/scripts/zcat_cross_framework.py --manifest experiments/configs/zcat_cross_framework_smoke.yaml validate
python experiments/scripts/zcat_cross_framework.py --manifest experiments/configs/zcat_cross_framework_smoke.yaml run --resume
python experiments/scripts/zcat_cross_framework.py --manifest experiments/configs/zcat_cross_framework_smoke.yaml analyze
```

For the full campaign, switch to `experiments/configs/zcat_cross_framework.yaml`.

## Deadline

Both manifests include:

```yaml
deadline:
  stop_new_runs_at: "2026-02-19 09:00:00"
```

Behavior:
- The `run` command stops launching new tasks when local clock reaches that timestamp.
- Already finished tasks remain in `runs.csv`.
- `run_manifest_resolved.json` records whether the run stopped by deadline.

## Parallelism

Both manifests include:

```yaml
run:
  workers: -1
```

Meaning:
- `workers: -1` resolves to `cpu_count() - 1`.
- You can override from CLI with `run --workers N`.

## Outputs

For each manifest, results are written under `paths.output_dir`:

- `validation/precheck_objective_equivalence.csv`
- `validation/precheck_reference_fronts.csv`
- `runs.csv`
- `raw/.../front.csv` and `raw/.../meta.json`
- `analysis/metrics.csv`
- `analysis/summary.csv`
- `analysis/pairwise_wilcoxon.csv`
- `analysis/friedman.csv`
