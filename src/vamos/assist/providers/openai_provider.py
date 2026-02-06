from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Mapping
from typing import Any, cast

from vamos.experiment.cli.quickstart import generate_quickstart_config

from .protocol import ProviderResponse

_OPENAI_SETUP_HINT = (
    "OpenAI provider setup required: install dependencies with "
    "`pip install vamos[openai]` (or `pip install openai`), set OPENAI_API_KEY, "
    "and run `vamos assist doctor` for diagnostics."
)
_DEFAULT_MODEL = "gpt-5.2"
_DEFAULT_TEMPERATURE = 0.2
_DEFAULT_MAX_OUTPUT_TOKENS = 900
_PROBLEM_TYPES: tuple[str, ...] = ("real", "int", "binary")
_LIST_LIMIT = 50
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS: tuple[float, ...] = (0.5, 1.0, 2.0)
_TRANSIENT_ERROR_HINTS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "rate limit",
    "too many requests",
    "temporary",
    "temporarily",
    "try again",
    "connection reset",
    "connection aborted",
    "service unavailable",
    "unavailable",
    "429",
    "503",
)


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _allowed_override_keys(templates: list[str]) -> list[str]:
    allowed: set[str] = set()
    for template in templates:
        try:
            config = generate_quickstart_config(template=template, overrides=None)
        except Exception:
            continue
        defaults = config.get("defaults")
        if isinstance(defaults, Mapping):
            for key in defaults:
                if isinstance(key, str):
                    allowed.add(key)
    return sorted(allowed)


def _as_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _truncate_list(items: list[str], *, limit: int = _LIST_LIMIT) -> tuple[list[str], bool]:
    if len(items) <= limit:
        return list(items), False
    return list(items[:limit]), True


def _build_compact_catalog_context(catalog: Mapping[str, object]) -> tuple[dict[str, object], list[str]]:
    context: dict[str, object] = {}
    notes: list[str] = []

    for key in ("algorithms", "kernels"):
        values = _as_string_list(catalog.get(key))
        truncated_values, was_truncated = _truncate_list(values)
        context[key] = truncated_values
        if was_truncated:
            notes.append(f"{key} truncated to first {_LIST_LIMIT} of {len(values)} entries.")

    for key in ("crossover_methods", "mutation_methods"):
        values = _as_string_list(catalog.get(key))
        if not values:
            continue
        if len(values) <= _LIST_LIMIT:
            context[key] = values
            continue
        preview, _ = _truncate_list(values)
        context[f"{key}_preview"] = preview
        notes.append(f"{key} preview truncated to first {_LIST_LIMIT} of {len(values)} entries.")

    return context, notes


def _response_schema(templates: list[str]) -> dict[str, object]:
    return {
        "type": "object",
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "const": "plan"},
                    "template": {"type": "string", "enum": templates},
                    "problem_type": {"type": "string", "enum": list(_PROBLEM_TYPES)},
                    "overrides": {"type": "object", "additionalProperties": True},
                    "warnings": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["kind", "template", "problem_type"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "const": "questions"},
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 3,
                    },
                    "warnings": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["kind", "questions"],
            },
        ],
    }


def _coerce_provider_response(raw: object) -> ProviderResponse:
    if not isinstance(raw, dict):
        raise RuntimeError("OpenAI response is not a JSON object.")
    kind = raw.get("kind")
    if kind == "plan":
        template = raw.get("template")
        problem_type = raw.get("problem_type")
        if not isinstance(template, str) or not isinstance(problem_type, str):
            raise RuntimeError("OpenAI plan response is missing template/problem_type.")
        return cast(ProviderResponse, raw)
    if kind == "questions":
        questions = raw.get("questions")
        if not isinstance(questions, list) or any(not isinstance(item, str) for item in questions):
            raise RuntimeError("OpenAI questions response is missing a valid questions list.")
        return cast(ProviderResponse, raw)
    raise RuntimeError("OpenAI response kind must be 'plan' or 'questions'.")


class OpenAIPlanProvider:
    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self.model = model or os.getenv("VAMOS_ASSIST_OPENAI_MODEL", _DEFAULT_MODEL)
        self.temperature = (
            temperature if temperature is not None else _read_float_env("VAMOS_ASSIST_OPENAI_TEMPERATURE", _DEFAULT_TEMPERATURE)
        )
        self.max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else _read_int_env("VAMOS_ASSIST_OPENAI_MAX_OUTPUT_TOKENS", _DEFAULT_MAX_OUTPUT_TOKENS)
        )

    def _import_openai(self) -> Any:
        from openai import OpenAI  # type: ignore[import-not-found]

        return OpenAI

    def _get_client(self) -> Any:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(_OPENAI_SETUP_HINT)
        try:
            openai_cls = self._import_openai()
        except ImportError as exc:
            raise RuntimeError(_OPENAI_SETUP_HINT) from exc
        return openai_cls()

    def _responses_create(self, client: Any, **kwargs: object) -> Any:
        return client.responses.create(**kwargs)

    def _is_transient_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return any(token in message for token in _TRANSIENT_ERROR_HINTS)

    def _create_response_with_retry(self, client: Any, **kwargs: object) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return self._responses_create(client, **kwargs)
            except Exception as exc:
                last_error = exc
                if attempt >= _MAX_ATTEMPTS or not self._is_transient_error(exc):
                    break
                delay = _BACKOFF_SECONDS[min(attempt - 1, len(_BACKOFF_SECONDS) - 1)] + random.uniform(0.0, 0.1)
                time.sleep(delay)

        if last_error is None:  # pragma: no cover - defensive
            raise RuntimeError("OpenAI provider failed unexpectedly. Run `vamos assist doctor` for diagnostics.")
        raise RuntimeError(
            f"OpenAI provider failed after {_MAX_ATTEMPTS} attempts: {last_error}. Run `vamos assist doctor` for diagnostics."
        ) from last_error

    def propose(
        self,
        prompt: str,
        catalog: Mapping[str, object],
        templates: list[str],
        problem_type_hint: str | None = None,
        answers: dict[str, str] | None = None,
    ) -> ProviderResponse:
        if not templates:
            raise RuntimeError("OpenAI provider failed: no templates available.")
        schema = _response_schema(templates)
        allowed_keys = _allowed_override_keys(templates)
        compact_catalog, catalog_notes = _build_compact_catalog_context(catalog)
        allowed_keys_limited, allowed_keys_truncated = _truncate_list(allowed_keys)
        if allowed_keys_truncated:
            catalog_notes.append(f"allowed_override_keys truncated to first {_LIST_LIMIT} of {len(allowed_keys)} entries.")

        system_prompt = (
            "You are the VAMOS assist planner.\n"
            "Return exactly one JSON object that matches the provided schema.\n"
            "Choose the best template from the provided list.\n"
            "Use problem_type in {real,int,binary}.\n"
            "If you have enough information, return kind='plan'.\n"
            "If clarifications are necessary, return kind='questions' with at most 3 short questions.\n"
            "Keep overrides minimal and only use keys listed in allowed_override_keys."
        )
        user_payload: dict[str, object] = {
            "prompt": prompt,
            "problem_type_hint": problem_type_hint,
            "templates_count": len(templates),
            "catalog": compact_catalog,
            "allowed_override_keys": allowed_keys_limited,
            "notes": catalog_notes,
        }
        if answers:
            user_payload["answers"] = answers
        user_message = json.dumps(user_payload, indent=2, sort_keys=True)

        try:
            client = self._get_client()
            response = self._create_response_with_retry(
                client,
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "vamos_plan",
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
            raw = getattr(response, "output_text", None)
            if not isinstance(raw, str) or not raw.strip():
                raise RuntimeError("No output_text in OpenAI response.")
            parsed = json.loads(raw)
            return _coerce_provider_response(parsed)
        except RuntimeError as exc:
            if _OPENAI_SETUP_HINT in str(exc):
                raise
            if str(exc).startswith("OpenAI provider failed:"):
                raise
            raise RuntimeError(f"OpenAI provider failed: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"OpenAI provider failed: {exc}") from exc


__all__ = ["OpenAIPlanProvider"]
