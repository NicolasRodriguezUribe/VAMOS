"""Distribution, audit, SBOM, and checksum support for VAMOS releases."""

from __future__ import annotations

import email.parser
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import venv
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from release_policy import scan_files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distributions(directory: Path) -> tuple[Path, Path]:
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise AssertionError(f"Expected one wheel and one sdist in {directory}; found {wheels} and {sdists}.")
    return wheels[0], sdists[0]


def copy_distributions(source: Path, destination: Path) -> tuple[Path, Path]:
    wheel, sdist = distributions(source)
    destination.mkdir(parents=True, exist_ok=False)
    copied = (destination / wheel.name, destination / sdist.name)
    shutil.copyfile(wheel, copied[0])
    shutil.copyfile(sdist, copied[1])
    if [sha256_file(item) for item in copied] != [sha256_file(wheel), sha256_file(sdist)]:
        raise AssertionError("Copied release artifacts differ from their frozen inputs.")
    return copied


def inspect_distributions(directory: Path, version: str) -> dict[str, Any]:
    wheel, sdist = distributions(directory)
    wheel_names, wheel_payloads = _zip_members(wheel)
    sdist_names, sdist_payloads = _tar_members(sdist)
    _safe_member_names(wheel_names, wheel.name)
    _safe_member_names(sdist_names, sdist.name)
    metadata_name = _single((name for name in wheel_names if name.endswith(".dist-info/METADATA")), "wheel METADATA")
    entry_name = _single((name for name in wheel_names if name.endswith(".dist-info/entry_points.txt")), "wheel entry points")
    wheel_metadata = _metadata(wheel_payloads[metadata_name])
    pkg_info_name = _single((name for name in sdist_names if name.endswith("/PKG-INFO") and name.count("/") == 1), "sdist root PKG-INFO")
    sdist_metadata = _metadata(sdist_payloads[pkg_info_name])
    required_wheel = {
        "vamos/py.typed": "py.typed",
        entry_name: "entry points",
    }
    missing_wheel = [label for name, label in required_wheel.items() if name not in wheel_names]
    license_names = [name for name in wheel_names if name.endswith(".dist-info/licenses/LICENSE")]
    required_sdist_suffixes = ("/LICENSE", "/README.md", "/pyproject.toml", "/src/vamos/py.typed")
    missing_sdist = [suffix for suffix in required_sdist_suffixes if not any(name.endswith(suffix) for name in sdist_names)]
    if missing_wheel or not license_names or missing_sdist:
        raise AssertionError(f"Distribution content is incomplete: wheel={missing_wheel}, license={license_names}, sdist={missing_sdist}")
    if "vamos = vamos.experiment.cli.main:main" not in wheel_payloads[entry_name].decode("utf-8"):
        raise AssertionError("Stable 'vamos' console entry point is missing from the wheel.")
    for source, metadata in (("wheel", wheel_metadata), ("sdist", sdist_metadata)):
        expected = {
            "Name": "vamos-optimization",
            "Version": version,
            "License-Expression": "MIT",
        }
        mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
        if mismatches:
            raise AssertionError(f"{source} metadata mismatch: {mismatches}")
        if "LICENSE" not in metadata.get_all("License-File", []):
            raise AssertionError(f"{source} metadata does not declare LICENSE.")
    if any(name.endswith((".pyc", ".pyo")) or "__pycache__" in name for name in wheel_names + sdist_names):
        raise AssertionError("Bytecode/cache content was packaged.")
    duplicate_groups = _duplicate_payload_groups(wheel_payloads)
    unexpected_duplicates = [
        group for group in duplicate_groups if not all(name.startswith("vamos/resources/reference_fronts/") for name in group)
    ]
    if unexpected_duplicates:
        raise AssertionError(f"Unexpected duplicate wheel payloads: {unexpected_duplicates[:5]}")
    _scan_archive_payloads(wheel_payloads, wheel.name)
    _scan_archive_payloads(sdist_payloads, sdist.name)
    return {
        "wheel": {"name": wheel.name, "size": wheel.stat().st_size, "sha256": sha256_file(wheel)},
        "sdist": {"name": sdist.name, "size": sdist.stat().st_size, "sha256": sha256_file(sdist)},
        "metadata": {
            "name": wheel_metadata["Name"],
            "version": wheel_metadata["Version"],
            "license_expression": wheel_metadata["License-Expression"],
            "requires_python": wheel_metadata["Requires-Python"],
        },
        "wheel_files": len(wheel_names),
        "sdist_files": len(sdist_names),
        "duplicate_reference_front_groups": duplicate_groups,
    }


def create_runtime_environment(root: Path, wheel: Path, constraints: Path, *, extras: str = "compute") -> tuple[Path, Path]:
    if root.exists():
        raise FileExistsError(f"Refusing to reuse release environment: {root}")
    venv.EnvBuilder(with_pip=True).create(root)
    python = environment_python(root)
    requirement = f"{wheel.resolve()}[{extras}]" if extras else str(wheel.resolve())
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    installed = subprocess.run(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "-c", str(constraints), requirement],
        capture_output=True,
        env=environment,
        text=True,
        check=False,
    )
    if installed.returncode != 0:
        raise RuntimeError(installed.stderr.strip() or installed.stdout.strip())
    checked = subprocess.run([str(python), "-m", "pip", "check"], capture_output=True, env=environment, text=True, check=False)
    if checked.returncode != 0:
        raise RuntimeError(checked.stderr.strip() or checked.stdout.strip())
    return root, python


def environment_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def write_runtime_lock(python: Path, output: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    code = (
        "import importlib.metadata as m, json; "
        "skip={'pip','setuptools','wheel','vamos-optimization'}; "
        "items=sorted((d.metadata['Name'], d.version) for d in m.distributions() "
        "if d.metadata.get('Name','').lower() not in skip); "
        "print(json.dumps(items))"
    )
    completed = subprocess.run([str(python), "-c", code], capture_output=True, env=environment, text=True, check=True)
    items = json.loads(completed.stdout)
    output.write_text("".join(f"{name}=={version}\n" for name, version in items), encoding="utf-8", newline="\n")
    return {"requirements": len(items), "path": output.name}


def run_dependency_audit(python: Path, requirements: Path, output: Path, *, blocking: bool) -> dict[str, Any]:
    completed = subprocess.run(
        [str(python), "-m", "pip_audit", "-r", str(requirements), "--format", "json", "--output", str(output)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if not output.is_file():
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "pip-audit produced no report")
    payload = json.loads(output.read_text(encoding="utf-8"))
    dependencies = payload.get("dependencies", []) if isinstance(payload, dict) else payload
    finding_ids = {
        (str(item.get("name")), str(vulnerability.get("id")))
        for item in dependencies
        if isinstance(item, dict)
        for vulnerability in item.get("vulns", [])
        if isinstance(vulnerability, dict)
    }
    findings = len(finding_ids)
    if completed.returncode not in (0, 1):
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"pip-audit exited {completed.returncode}")
    if blocking and findings:
        raise AssertionError(f"Runtime dependency audit reported {findings} unresolved vulnerability record(s).")
    return {
        "requirements": str(requirements),
        "dependencies": len(dependencies),
        "findings": findings,
        "blocking": blocking,
        "exit_code": completed.returncode,
    }


def create_sbom(tool_python: Path, runtime_lock: Path, wheel: Path, version: str, output: Path) -> dict[str, Any]:
    executable = tool_python.with_name("cyclonedx-py.exe" if os.name == "nt" else "cyclonedx-py")
    completed = subprocess.run(
        [
            str(executable),
            "requirements",
            str(runtime_lock),
            "--output-format",
            "JSON",
            "--output-file",
            str(output),
            "--mc-type",
            "library",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    payload = json.loads(output.read_text(encoding="utf-8"))
    component = {
        "bom-ref": f"pkg:pypi/vamos-optimization@{version}",
        "type": "library",
        "name": "vamos-optimization",
        "version": version,
        "purl": f"pkg:pypi/vamos-optimization@{version}",
        "hashes": [{"alg": "SHA-256", "content": sha256_file(wheel)}],
    }
    components = payload.setdefault("components", [])
    components[:] = [item for item in components if item.get("name") != "vamos-optimization"]
    components.append(component)
    components.sort(key=lambda item: (str(item.get("name", "")).lower(), str(item.get("version", ""))))
    payload.setdefault("metadata", {})["component"] = component
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    scan_files([output], root=output.parent)
    return {"path": output.name, "sha256": sha256_file(output), "components": len(components), "format": "CycloneDX JSON"}


def write_release_manifests(output: Path, dist: Path, sbom: Path, version: str, commit: str) -> dict[str, Any]:
    wheel, sdist = distributions(dist)
    files = [
        (f"dist/{wheel.name}", wheel),
        (f"dist/{sdist.name}", sdist),
        (sbom.name, sbom),
    ]
    checksums = output / "SHA256SUMS"
    checksums.write_text("".join(f"{sha256_file(path)}  {logical}\n" for logical, path in files), encoding="utf-8", newline="\n")
    provenance = output / "build-provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "document_type": "vamos.build-provenance",
                "repository": "vamos-optimization/VAMOS",
                "schema_version": "1.0.0",
                "project": "vamos-optimization",
                "version": version,
                "source_commit": commit,
                "artifacts": [logical for logical, _ in files[:2]],
                "builder": {"python": sys.version.split()[0], "platform": sys.platform},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = output / "artifact-manifest.json"
    manifest_files = [{"filename": logical, "size": path.stat().st_size, "sha256": sha256_file(path)} for logical, path in files]
    manifest_files.extend(
        {
            "filename": path.name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (checksums, provenance)
    )
    manifest.write_text(
        json.dumps(
            {
                "document_type": "vamos.release-artifact-manifest",
                "repository": "vamos-optimization/VAMOS",
                "schema_version": "1.0.0",
                "project": "vamos-optimization",
                "version": version,
                "source_commit": commit,
                "files": manifest_files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "checksums": checksums.name,
        "checksums_sha256": sha256_file(checksums),
        "provenance": provenance.name,
        "provenance_sha256": sha256_file(provenance),
        "manifest": manifest.name,
        "manifest_sha256": sha256_file(manifest),
    }


def _zip_members(path: Path) -> tuple[list[str], dict[str, bytes]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        return names, {name: archive.read(name) for name in names if not name.endswith("/")}


def _tar_members(path: Path) -> tuple[list[str], dict[str, bytes]]:
    with tarfile.open(path, mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        payloads: dict[str, bytes] = {}
        for member in members:
            stream = archive.extractfile(member)
            if stream is not None:
                payloads[member.name] = stream.read()
        return [member.name for member in members], payloads


def _safe_member_names(names: list[str], archive: str) -> None:
    unsafe = []
    for name in names:
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or "\\" in name:
            unsafe.append(name)
    if unsafe:
        raise AssertionError(f"Unsafe member names in {archive}: {unsafe[:10]}")


def _metadata(payload: bytes) -> Any:
    return email.parser.BytesParser().parsebytes(payload)


def _single(values: Any, label: str) -> str:
    items = list(values)
    if len(items) != 1:
        raise AssertionError(f"Expected one {label}; found {items}")
    return str(items[0])


def _scan_archive_payloads(payloads: dict[str, bytes], archive_name: str) -> None:
    with _ArchiveScanRoot(archive_name, payloads) as paths:
        scan_files(paths, root=paths[0].parent if paths else Path.cwd())


def _duplicate_payload_groups(payloads: dict[str, bytes]) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for name, payload in payloads.items():
        if not payload:
            continue
        groups.setdefault(hashlib.sha256(payload).hexdigest(), []).append(name)
    return [sorted(names) for names in groups.values() if len(names) > 1]


class _ArchiveScanRoot:
    def __init__(self, archive_name: str, payloads: dict[str, bytes]) -> None:
        import tempfile

        self._temporary = tempfile.TemporaryDirectory(prefix="vamos-archive-scan-")
        self.root = Path(self._temporary.name)
        self.archive_name = archive_name
        self.payloads = payloads

    def __enter__(self) -> list[Path]:
        paths: list[Path] = []
        for index, (name, payload) in enumerate(self.payloads.items()):
            path = self.root / f"{index:05d}-{Path(name).name}"
            path.write_bytes(payload)
            paths.append(path)
        return paths

    def __exit__(self, *_args: object) -> None:
        self._temporary.cleanup()


__all__ = [
    "copy_distributions",
    "create_runtime_environment",
    "create_sbom",
    "distributions",
    "environment_python",
    "inspect_distributions",
    "run_dependency_audit",
    "sha256_file",
    "write_release_manifests",
    "write_runtime_lock",
]
