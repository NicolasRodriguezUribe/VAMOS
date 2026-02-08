from __future__ import annotations

import json

from vamos.assist.doctor import collect_doctor_report, format_doctor_report_text


def test_collect_doctor_report_is_json_serializable_and_has_expected_keys() -> None:
    report = collect_doctor_report()
    json.dumps(report, sort_keys=True)

    for key in ("python", "platform", "vamos", "catalog_summary", "openai"):
        assert key in report

    python_info = report["python"]
    platform_info = report["platform"]
    vamos_info = report["vamos"]
    catalog_summary = report["catalog_summary"]
    openai_info = report["openai"]

    assert isinstance(python_info, dict)
    assert isinstance(platform_info, dict)
    assert isinstance(vamos_info, dict)
    assert isinstance(catalog_summary, dict)
    assert isinstance(openai_info, dict)

    assert "version" in python_info
    assert "executable" in python_info

    for key in ("system", "release", "machine"):
        assert key in platform_info

    assert "version" in vamos_info

    for key in ("templates_count", "algorithms_count", "kernels_count", "kernels", "templates_preview"):
        assert key in catalog_summary

    assert isinstance(catalog_summary["kernels"], list)
    assert isinstance(catalog_summary["templates_preview"], list)

    for key in ("sdk_available", "api_key_set", "model", "temperature", "max_output_tokens"):
        assert key in openai_info
    assert isinstance(openai_info["sdk_available"], bool)
    assert isinstance(openai_info["api_key_set"], bool)


def test_format_doctor_report_text_returns_readable_summary() -> None:
    report = collect_doctor_report()
    text = format_doctor_report_text(report)
    assert isinstance(text, str)
    assert "VAMOS Assist Doctor" in text
    assert "Python:" in text
    assert "Platform:" in text
    assert "OpenAI:" in text
