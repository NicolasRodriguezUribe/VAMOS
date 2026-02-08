from __future__ import annotations

import importlib
import os
import platform
import sys
from typing import Any

from .catalog import build_catalog
from .providers.openai_provider import OpenAIPlanProvider


def _vamos_version() -> str | None:
    try:
        from vamos.foundation.version import get_version

        return str(get_version())
    except Exception:
        return None


def _openai_sdk_available() -> bool:
    try:
        importlib.import_module("openai")
    except Exception:
        return False
    return True


def _templates_preview(templates: list[str], limit: int = 20) -> list[str]:
    preview = list(templates[:limit])
    if len(templates) > limit:
        preview.append("…")
    return preview


def collect_doctor_report() -> dict[str, Any]:
    catalog = build_catalog(problem_type="real")
    templates = list(catalog.get("templates", []))
    kernels = list(catalog.get("kernels", []))
    algorithms = list(catalog.get("algorithms", []))
    openai_defaults = OpenAIPlanProvider()

    report: dict[str, Any] = {
        "python": {
            "version": sys.version.replace("\n", " "),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "vamos": {
            "version": _vamos_version(),
        },
        "catalog_summary": {
            "templates_count": len(templates),
            "algorithms_count": len(algorithms),
            "kernels_count": len(kernels),
            "kernels": sorted(kernels),
            "templates_preview": _templates_preview(sorted(templates), limit=20),
        },
        "openai": {
            "sdk_available": _openai_sdk_available(),
            "api_key_set": bool(os.getenv("OPENAI_API_KEY", "").strip()),
            "model": openai_defaults.model,
            "temperature": openai_defaults.temperature,
            "max_output_tokens": openai_defaults.max_output_tokens,
        },
    }
    return report


def format_doctor_report_text(report: dict[str, Any]) -> str:
    python_info = report.get("python")
    platform_info = report.get("platform")
    vamos_info = report.get("vamos")
    catalog_summary = report.get("catalog_summary")
    openai_info = report.get("openai")

    lines: list[str] = ["VAMOS Assist Doctor"]

    if isinstance(python_info, dict):
        lines.append("Python:")
        lines.append(f"  version: {python_info.get('version')}")
        lines.append(f"  executable: {python_info.get('executable')}")

    if isinstance(platform_info, dict):
        lines.append("Platform:")
        lines.append(f"  system: {platform_info.get('system')}")
        lines.append(f"  release: {platform_info.get('release')}")
        lines.append(f"  machine: {platform_info.get('machine')}")

    if isinstance(vamos_info, dict):
        lines.append("VAMOS:")
        lines.append(f"  version: {vamos_info.get('version')}")

    if isinstance(catalog_summary, dict):
        lines.append("Catalog:")
        lines.append(f"  templates_count: {catalog_summary.get('templates_count')}")
        lines.append(f"  algorithms_count: {catalog_summary.get('algorithms_count')}")
        lines.append(f"  kernels_count: {catalog_summary.get('kernels_count')}")
        kernels = catalog_summary.get("kernels")
        if isinstance(kernels, list):
            lines.append(f"  kernels: {', '.join(str(item) for item in kernels)}")
        preview = catalog_summary.get("templates_preview")
        if isinstance(preview, list):
            lines.append(f"  templates_preview: {', '.join(str(item) for item in preview)}")

    if isinstance(openai_info, dict):
        lines.append("OpenAI:")
        lines.append(f"  sdk_available: {openai_info.get('sdk_available')}")
        lines.append(f"  api_key_set: {openai_info.get('api_key_set')}")
        lines.append(f"  model: {openai_info.get('model')}")
        lines.append(f"  temperature: {openai_info.get('temperature')}")
        lines.append(f"  max_output_tokens: {openai_info.get('max_output_tokens')}")

    return "\n".join(lines) + "\n"


__all__ = ["collect_doctor_report", "format_doctor_report_text"]
