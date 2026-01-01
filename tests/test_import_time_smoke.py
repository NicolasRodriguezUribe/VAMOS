from __future__ import annotations

import json
import subprocess
import sys


BLACKLIST = (
    "torch",
    "tensorflow",
    "jax",
    "sklearn",
    "pymoo",
    "jmetalpy",
    "pygmo",
    "streamlit",
)


def _collect_imported_modules() -> list[str]:
    cmd = [
        sys.executable,
        "-c",
        "\n".join(
            [
                "import json, sys",
                "mods = ['vamos', 'vamos.api', 'vamos.engine.api', 'vamos.experiment.quick', 'vamos.ux.api']",
                "for m in mods:",
                "    __import__(m)",
                "print(json.dumps(sorted(sys.modules.keys())))",
            ]
        ),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(f"Import-time smoke failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        raise AssertionError("Import-time smoke produced no output.")
    payload = stdout_lines[-1]
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"Failed to parse sys.modules payload:\n{payload}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        ) from exc


def _find_blacklisted_modules(modules: list[str]) -> list[str]:
    offenders = []
    for banned in BLACKLIST:
        for name in modules:
            if name == banned or name.startswith(f"{banned}."):
                offenders.append(name)
                break
    return sorted(set(offenders))


def test_import_time_smoke() -> None:
    modules = _collect_imported_modules()
    offenders = _find_blacklisted_modules(modules)
    if offenders:
        raise AssertionError(f"Import-time smoke loaded optional heavy modules: {offenders}")
