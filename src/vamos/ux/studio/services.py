from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


from vamos.engine.algorithm.config import (
    AGEMOEAConfig,
    GenericAlgorithmConfig,
    IBEAConfig,
    MOEADConfig,
    NSGAIIConfig,
    NSGAIIIConfig,
    RVEAConfig,
    SMPSOConfig,
    SMSEMOAConfig,
    SPEA2Config,
)
from vamos.engine.algorithm.catalog import DEFAULT_ALGORITHM
from vamos.engine.algorithm.config.base import _SerializableConfig
from vamos.engine.algorithm.config.defaults import build_default_algorithm_config
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.engine.algorithm.registry import get_algorithms_registry, resolve_algorithm
from vamos.exceptions import InvalidAlgorithmError

_CONFIG_MAP: dict[str, type[_SerializableConfig]] = {
    "nsgaii": NSGAIIConfig,
    "moead": MOEADConfig,
    "spea2": SPEA2Config,
    "smsemoa": SMSEMOAConfig,
    "nsgaiii": NSGAIIIConfig,
    "ibea": IBEAConfig,
    "smpso": SMPSOConfig,
    "agemoea": AGEMOEAConfig,
    "rvea": RVEAConfig,
}
from vamos.foundation.eval import EvaluationBackend
from vamos.foundation.eval.backends import resolve_eval_strategy
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.problem.types import ProblemProtocol
from vamos.ux.analysis.mcdm import reference_point_scores
from vamos.ux.studio.data import build_fronts, load_runs_from_study
from vamos.ux.studio.dm import build_decision_view

if TYPE_CHECKING:
    from vamos.ux.studio.data import FrontRecord, RunRecord
    from vamos.ux.studio.dm import DecisionView


class DynamicsCallback:
    def __init__(self) -> None:
        self.history: list[np.ndarray] = []

    def on_start(self, ctx: Any) -> None:
        pass

    def on_generation(self, generation: int, *, F: Any = None, **kwargs: Any) -> None:
        try:
            if F is not None:
                self.history.append(np.array(F))
        except Exception:
            _logger().debug("Failed to capture population snapshot in DynamicsCallback", exc_info=True)

    def on_end(self, *, final_F: Any = None) -> None:
        try:
            if final_F is not None:
                self.history.append(np.array(final_F))
        except Exception:
            _logger().debug("Failed to capture final front in DynamicsCallback", exc_info=True)

    def __call__(self, algorithm: Any) -> None:
        try:
            if hasattr(algorithm, "pop") and algorithm.pop is not None:
                F = algorithm.pop.get("F")
                if F is not None:
                    self.history.append(np.array(F))
        except Exception:
            _logger().debug("Failed to capture population snapshot in DynamicsCallback", exc_info=True)


def load_studio_data(study_dir: Path) -> tuple[list[RunRecord], list[FrontRecord]]:
    runs = load_runs_from_study(study_dir)
    fronts = build_fronts(runs)
    return runs, fronts


def _contains_result_files(path: Path) -> bool:
    try:
        next(path.rglob("FUN.csv"))
    except StopIteration:
        return False
    return True


def discover_study_directories(base_dir: Path, *, limit: int = 8) -> list[Path]:
    """Return a short list of likely study/result directories for the UI picker."""
    base_dir = base_dir.resolve()
    candidates: list[Path] = []

    def add_candidate(path: Path) -> None:
        path = path.resolve()
        if path in candidates or not path.exists() or not path.is_dir():
            return
        if not _contains_result_files(path):
            return
        candidates.append(path)

    preferred_roots = [
        base_dir / "results",
        base_dir / "results" / "quickstart",
        base_dir / "paper" / "results",
    ]
    for root in preferred_roots:
        add_candidate(root)
        if root.exists():
            for child in sorted(root.iterdir()):
                if child.is_dir():
                    add_candidate(child)

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[:limit]


def build_demo_study_data() -> tuple[list[RunRecord], list[FrontRecord]]:
    """Return a small built-in dataset so the Explore tab is useful on first launch."""
    from vamos.ux.studio.data import FrontRecord, RunRecord

    x = np.linspace(0.02, 1.0, 24)
    nsgaii_front = np.column_stack([x, 1.02 - np.sqrt(x)])
    moead_front = np.column_stack([x, 1.04 - np.power(x, 0.62)])
    nsgaii_front = np.clip(nsgaii_front, 0.0, None)
    moead_front = np.clip(moead_front, 0.0, None)
    points_x = np.column_stack([x, 1.0 - x])

    demo_runs = [
        RunRecord(
            suite_name="demo",
            experiment_id="demo/nsgaii",
            problem_name="Demo trade-off",
            algorithm_name="NSGA-II demo",
            seed=0,
            fun=nsgaii_front,
            var=points_x,
            metadata={"demo": True},
        ),
        RunRecord(
            suite_name="demo",
            experiment_id="demo/moead",
            problem_name="Demo trade-off",
            algorithm_name="MOEA/D demo",
            seed=1,
            fun=moead_front,
            var=points_x,
            metadata={"demo": True},
        ),
    ]
    demo_fronts = [
        FrontRecord(
            problem_name="Demo trade-off",
            algorithm_name="NSGA-II demo",
            points_F=nsgaii_front,
            points_X=points_x,
            extra={"demo": True, "seeds": [0], "config": None},
        ),
        FrontRecord(
            problem_name="Demo trade-off",
            algorithm_name="MOEA/D demo",
            points_F=moead_front,
            points_X=points_x,
            extra={"demo": True, "seeds": [1], "config": None},
        ),
    ]
    return demo_runs, demo_fronts


def build_decision_views(
    fronts: list[FrontRecord],
    weights: np.ndarray,
    reference_point: np.ndarray | None,
    method: str,
) -> list[DecisionView]:
    views = []
    for front in fronts:
        view = build_decision_view(front, weights=weights, reference_point=reference_point, methods=[method, "weighted_sum", "knee"])
        views.append(view)
    return views


def _with_result_mode(cfg_data: AlgorithmConfigProtocol, result_mode: str) -> AlgorithmConfigProtocol:
    fields_map = getattr(cfg_data, "__dataclass_fields__", None)
    if not isinstance(fields_map, dict):
        return cfg_data
    if "result_mode" not in fields_map:
        return cfg_data
    return cast(AlgorithmConfigProtocol, replace(cast(Any, cfg_data), result_mode=result_mode))


def _build_algorithm_config(
    algorithm: str,
    *,
    pop_size: int | None,
    n_var: int | None,
    n_obj: int | None,
    encoding: str | None = None,
) -> AlgorithmConfigProtocol:
    algorithm = algorithm.lower()
    result_mode = "population"

    default_cfg = build_default_algorithm_config(
        algorithm,
        pop_size=pop_size,
        n_var=n_var,
        n_obj=n_obj,
        encoding=encoding,
    )
    if default_cfg is not None:
        return _with_result_mode(default_cfg, result_mode)

    registry = get_algorithms_registry()
    if algorithm in registry:
        base: dict[str, object] = {}
        if pop_size is not None:
            base["pop_size"] = pop_size
        if n_var is not None:
            base["n_var"] = n_var
        if n_obj is not None:
            base["n_obj"] = n_obj
        return GenericAlgorithmConfig(base)

    available = sorted(registry.keys())
    raise InvalidAlgorithmError(algorithm, available=available)


def _run_algorithm(
    problem: ProblemProtocol,
    *,
    algorithm: str,
    algorithm_config: AlgorithmConfigProtocol,
    termination: tuple[str, object],
    seed: int,
    engine: str,
    eval_strategy: EvaluationBackend | str | None = None,
    live_viz: object | None = None,
) -> dict[str, Any]:
    cfg_dict = dict(algorithm_config.to_dict())
    if "engine" in cfg_dict:
        raise ValueError("engine must be configured via run arguments, not algorithm_config.")
    kernel = resolve_kernel(engine)

    if eval_strategy is not None:
        backend = resolve_eval_strategy(eval_strategy) if isinstance(eval_strategy, str) else eval_strategy
    else:
        backend_name = str(cfg_dict.get("eval_strategy", "serial"))
        backend = resolve_eval_strategy(backend_name)

    algo_ctor = resolve_algorithm(algorithm)
    algorithm_instance = algo_ctor(cfg_dict, kernel)
    run_fn = algorithm_instance.run
    result = run_fn(
        problem=problem,
        termination=termination,
        seed=seed,
        eval_strategy=backend,
        live_viz=live_viz,
    )
    return dict(result)


def run_focused_optimization(
    problem: str,
    reference_point: np.ndarray,
    algo: str,
    budget: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    selection = make_problem_selection(problem)
    instance = selection.instantiate()
    algo_name = algo or "nsgaii"
    algo_cfg = _build_algorithm_config(
        algo_name,
        pop_size=40,
        n_var=getattr(instance, "n_var", None),
        n_obj=getattr(instance, "n_obj", None),
        encoding=getattr(instance, "encoding", None),
    )
    result = _run_algorithm(
        instance,
        algorithm=algo_name,
        algorithm_config=algo_cfg,
        termination=("max_evaluations", budget),
        seed=0,
        engine="numpy",
    )
    F = result.get("F")
    if F is None:
        raise RuntimeError("Focused optimization returned no objectives.")

    scores = reference_point_scores(np.asarray(F), reference_point).scores
    order = np.argsort(scores)
    X = None
    if result.get("X") is not None:
        X = np.asarray(result["X"])[order]
    return np.asarray(F)[order], X


def run_with_history(
    problem_name: str,
    config: dict[str, Any],
    budget: int,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    selection = make_problem_selection(problem_name)
    problem = selection.instantiate()

    algo_name = str(config.get("algorithm", DEFAULT_ALGORITHM))
    algo_cfg_raw = config.get("algorithm_config", {})

    def _coerce_algo_config(cfg: Any) -> AlgorithmConfigProtocol:
        if not cfg:
            return _build_algorithm_config(
                algo_name,
                pop_size=100,
                n_var=getattr(problem, "n_var", None),
                n_obj=getattr(problem, "n_obj", None),
                encoding=getattr(problem, "encoding", None),
            )
        if isinstance(cfg, AlgorithmConfigProtocol):
            return cfg
        if not isinstance(cfg, dict):
            raise TypeError("algorithm_config must be a config object or dict.")

        config_cls = _CONFIG_MAP.get(algo_name.lower())
        if config_cls is None:
            raise TypeError(f"Dict coercion not supported for algorithm '{algo_name}'.")
        return cast(AlgorithmConfigProtocol, config_cls.from_dict(cfg))

    algo_cfg = _coerce_algo_config(algo_cfg_raw)
    callback = DynamicsCallback()

    result = _run_algorithm(
        problem,
        algorithm=algo_name,
        algorithm_config=algo_cfg,
        termination=("max_evaluations", budget),
        seed=int(config.get("seed", 0)),
        engine=str(config.get("engine", "numpy")),
        live_viz=callback,
    )
    return result, callback.history


_VAMOS_CODE_SYSTEM_PROMPT = """\
You are a code generator for the VAMOS multi-objective optimization framework.

The user will describe an optimization problem in natural language.  You must
return a JSON object with these fields:

  objective_code : str — Raw Python statements (NOT a function def).
      The variable `x` is a 1-D NumPy array of length n_var.
      Must end with `return [f0, f1, ...]` (one value per objective).
      Allowed imports: `math`, `numpy` (as np).

  constraint_code : str — Same format but for constraints (may be empty "").
      Convention: g(x) <= 0 is feasible.
      E.g. for stress <= 100: `return [stress - 100.0]`

  n_var : int — Number of decision variables.
  n_obj : int — Number of objectives (>= 2).
  bounds : str — Bounds as "lo, hi" per variable, one pair per line,
      or a single line if all variables share the same bounds.
      Example: "0.0, 5.0\\n1.0, 10.0"

Example — beam design (minimise cost and deflection, stress <= 100):

{
  "objective_code": "area = x[0] * x[1]\\ncost = 2.0 * x[0] + 3.0 * x[1]\\ndeflection = 1000.0 / (x[0] * x[1] ** 3 + 1e-6)\\nreturn [cost, deflection]",
  "constraint_code": "stress = 600.0 / (x[0] * x[1] ** 2 + 1e-6)\\nreturn [stress - 100.0]",
  "n_var": 2,
  "n_obj": 2,
  "bounds": "0.5, 5.0\\n1.0, 10.0"
}

Return ONLY the JSON object, nothing else.
"""


def _llm_generate_gemini(description: str, *, api_key: str = "") -> dict[str, Any]:
    """Call Google Gemini to generate VAMOS problem code."""
    import json
    import os

    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("No Gemini API key provided. Paste it in the API Key field or set GEMINI_API_KEY.")
    try:
        from google import genai  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "google-genai package not installed. Run: pip install google-genai"
        ) from exc

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=os.getenv("VAMOS_ASSIST_GEMINI_MODEL", "gemini-2.0-flash"),
        contents=f"{_VAMOS_CODE_SYSTEM_PROMPT}\n\nUser request:\n{description}",
    )
    raw = response.text
    if not raw or not raw.strip():
        raise RuntimeError("Empty response from Gemini.")
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return json.loads(text.strip())


def _llm_generate_openai(description: str) -> dict[str, Any]:
    """Call OpenAI to generate VAMOS problem code from a natural-language description."""
    import json
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "openai package not installed. Run: pip install vamos-optimization[openai]"
        ) from exc

    client = OpenAI()
    response = client.responses.create(
        model=os.getenv("VAMOS_ASSIST_OPENAI_MODEL", "gpt-4o"),
        input=[
            {"role": "system", "content": _VAMOS_CODE_SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ],
        temperature=0.2,
        max_output_tokens=1500,
    )
    raw = getattr(response, "output_text", None)
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError("Empty response from OpenAI.")
    return json.loads(raw)


def _llm_generate_anthropic(description: str) -> dict[str, Any]:
    """Call Anthropic Claude to generate VAMOS problem code."""
    import json
    import os

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
    try:
        from anthropic import Anthropic  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "anthropic package not installed. Run: pip install vamos-optimization[anthropic]"
        ) from exc

    client = Anthropic()
    message = client.messages.create(
        model=os.getenv("VAMOS_ASSIST_ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=1500,
        system=_VAMOS_CODE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": description}],
    )
    raw = message.content[0].text
    if not raw.strip():
        raise RuntimeError("Empty response from Anthropic.")
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return json.loads(text)


def llm_generate_problem_code(
    description: str,
    provider: str = "openai",
    api_key: str = "",
) -> dict[str, Any]:
    """Generate VAMOS problem code from a natural-language description using an LLM.

    Returns a dict with keys: objective_code, constraint_code, n_var, n_obj, bounds.
    """
    if provider == "gemini":
        result = _llm_generate_gemini(description, api_key=api_key)
    elif provider == "anthropic":
        result = _llm_generate_anthropic(description)
    else:
        result = _llm_generate_openai(description)

    # Validate required keys
    for key in ("objective_code", "n_var", "n_obj", "bounds"):
        if key not in result:
            raise RuntimeError(f"LLM response missing required field: {key}")
    result.setdefault("constraint_code", "")
    return result


__all__ = [
    "DynamicsCallback",
    "load_studio_data",
    "discover_study_directories",
    "build_demo_study_data",
    "build_decision_views",
    "run_focused_optimization",
    "run_with_history",
    "llm_generate_problem_code",
]
