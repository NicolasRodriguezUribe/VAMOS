# Release Smoke Verification

Use this checklist to verify source distribution and wheel packaging before a release.

## Build smoke (wheel/sdist)

```bash
python -m build
python -m pip install dist/*.whl
python -c "import vamos; print('ok')"
vamos assist doctor --json
```

## Recommended workflow

Use a clean virtual environment and clear previous build artifacts first:

```bash
python -m venv .venv-release-smoke
source .venv-release-smoke/bin/activate  # Windows PowerShell: .\.venv-release-smoke\Scripts\Activate.ps1
rm -rf build dist *.egg-info  # Windows PowerShell: Remove-Item -Recurse -Force build,dist; Remove-Item *.egg-info
```

Then run the build smoke commands above.

## Optional extras check

After installing the wheel, you can additionally verify OpenAI optional dependencies:

```bash
python -m pip install vamos-optimization[openai]
```

If working from a published package rather than local wheel testing:

```bash
python -m pip install vamos-optimization[openai]
```

## Optional automation script

You can run:

```bash
python scripts/verify_build_smoke.py
```

The script performs:

1. `python -m build`
2. create temporary virtual environment
3. install the newest wheel from `dist/`
4. `python -c "import vamos; print('ok')"`
5. `python -m vamos.experiment.cli.main assist doctor --json`
