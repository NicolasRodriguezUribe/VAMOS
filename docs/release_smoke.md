# Release Smoke Verification

Use this checklist to verify source distribution and wheel packaging before a release.

Release validation starts with the stable-facade and composite release gates:

```bash
python tools/typecheck.py --scope stable
python tools/typecheck.py --scope release
python tools/typecheck.py --scope full-zero  # informational for VAMOS 1.0.0
```

Do not build or publish while the stable or release command is nonzero.
`full-zero` currently reports inherited debt and is not a VAMOS 1.0 blocker.
Release success means strict/stable zero, an exact no-regression ratchet, and
health; it is not a claim of global typing cleanliness.

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
5. `vamos assist doctor --json`
