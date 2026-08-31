# Changing studies

The current study layer executes an in-memory sequence of `StudyTask` objects and returns `StudyResult` summaries. It does not yet provide durable study state, resume/retry, or a StudyManifest. The future behavior is frozen by the [durable study and StudyManifest v1 contract](study_manifest_contract.md), its [SA-001 through SA-074 acceptance specification](study_manifest_acceptance_tests.md), and [ADR 0008](adr/0008-durable-study-manifest-contract.md); those documents specify planned behavior, not an available production API.

## Current API

```python
from vamos.experiment.study import StudyTask
from vamos.experiment.study.api import run_study


tasks = [
    StudyTask(
        problem="zdt1",
        algorithm="nsgaii",
        engine="numpy",
        seed=seed,
        config_overrides={"population_size": 20, "max_evaluations": 100},
    )
    for seed in (1, 2, 3)
]
results = run_study(tasks)
```

`StudyRunner()` is constructed with verbosity, indicator, persister, evaluator, and termination options. Its `run(tasks, run_single_fn=...)` method requires the execution callable; `run_study(...)` is the supported convenience route that supplies the current single-run service.

`StudyResult.to_row()` is a derived summary. `CSVPersister.save_results(...)` may export those rows, but it does not own or duplicate canonical per-run arrays/specifications.

## Change procedure

1. Keep task definition in `types.py`, batch execution in `runner.py`, supported orchestration in `api.py`, and derived export in `persistence.py`.
2. Preserve input task order, explicit seeds, config override copying, encoding/problem resolution, and optional indicator behavior.
3. Route each persisted run through the canonical run-artifact writer. A study-level record may reference run/task identities; it must not mirror their specifications or arrays.
4. Distinguish execution failure policy from indicator/export failure. Never report a partially executed study as complete without explicit status.
5. Update the ablation API and CLI together when their use of study tasks changes.
6. Implement durable state, restart policy, retries, and resumability only through the frozen StudyManifest contract and its ordered roadmap, never as an incidental CSV extension.

## Required validation

```bash
python -m pytest -q tests/experiment/test_ablation_study_api.py tests/experiment/test_cli_ablation.py
python -m pytest -q tests/foundation/test_moocore_indicators.py tests/engine/test_hyperheuristic_indicators.py
```

Add focused runner/persistence tests for the changed behavior and run the full tier from `/AGENTS.md`.

```agent-docs
path: src/vamos/experiment/study/types.py
path: src/vamos/experiment/study/runner.py
path: src/vamos/experiment/study/api.py
path: src/vamos/experiment/study/persistence.py
path: tests/experiment/test_ablation_study_api.py
path: tests/experiment/test_cli_ablation.py
path: tests/foundation/test_moocore_indicators.py
path: tests/engine/test_hyperheuristic_indicators.py
path: docs/dev/study_manifest_contract.md
path: docs/dev/study_manifest_acceptance_tests.md
path: docs/dev/study_manifest_examples/README.md
path: docs/dev/adr/0008-durable-study-manifest-contract.md
symbol: vamos.experiment.study:StudyTask
symbol: vamos.experiment.study:StudyRunner
symbol: vamos.experiment.study.api:run_study
symbol: vamos.experiment.study.persistence:CSVPersister
command: python -m pytest -q tests/experiment/test_ablation_study_api.py tests/experiment/test_cli_ablation.py
command: python -m pytest -q tests/foundation/test_moocore_indicators.py tests/engine/test_hyperheuristic_indicators.py
```
