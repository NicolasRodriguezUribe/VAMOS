# VAMOS 1.0 compatibility baseline

These files are the permanent machine-readable baseline for the stable VAMOS
1.0 Python API, CLI, exit codes, algorithm configuration, and artifact schemas.
Run and study artifacts generated from the final 1.0.0 package are added beside
these snapshots during final release integration.

Regenerate the structural snapshots only through:

```bash
python tools/generate_v1_compatibility_snapshots.py
```

Changing an existing snapshot requires explicit compatibility review. Future
1.x releases must continue to pass the corresponding checks and load and
inspect the committed 1.0.0 artifact fixtures. Pre-1.0 artifacts do not belong
in this directory.
