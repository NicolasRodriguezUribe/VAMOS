from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .experiment import TunerKind, TuningExperiment, ExperimentResult
from .scenario import Scenario
from .sampler import Sampler, UniformSampler, ModelBasedSampler
from .validation import BenchmarkSuite, ConfigSpec
from .history import load_top_k_as_config_specs
from .tuning_task import TuningTask, EvalContext


@dataclass
class SamplerSpec:
    """
    Specification of the sampler used by the tuning experiment.
    """

    type: str = "uniform"
    exploration_prob: float = 0.2
    min_samples_to_model: int = 5

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "SamplerSpec":
        t = data.get("type", "uniform")
        return SamplerSpec(
            type=str(t),
            exploration_prob=float(data.get("exploration_prob", 0.2)),
            min_samples_to_model=int(data.get("min_samples_to_model", 5)),
        )


@dataclass
class RandomTunerSpec:
    """
    Specification for the basic random tuner.
    """

    max_trials: int = 50

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "RandomTunerSpec":
        return RandomTunerSpec(
            max_trials=int(data.get("max_trials", 50)),
        )


@dataclass
class RacingTunerSpec:
    """
    Specification for the racing tuner (RacingTuner).
    """

    max_initial_configs: int = 20
    scenario_params: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "RacingTunerSpec":
        max_initial = int(data.get("max_initial_configs", 20))
        scenario_data = data.get("scenario", {}) or {}
        if not isinstance(scenario_data, Mapping):
            raise ValueError("'racing.scenario' must be a mapping")
        return RacingTunerSpec(
            max_initial_configs=max_initial,
            scenario_params=dict(scenario_data),
        )


@dataclass
class BaselineSpec:
    """
    Explicit baseline configuration given directly in the experiment JSON.
    """

    label: str
    config: Dict[str, Any]

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "BaselineSpec":
        label = str(data.get("label", "baseline"))
        cfg = data.get("config", {}) or {}
        if not isinstance(cfg, Mapping):
            raise ValueError("BaselineSpec.config must be a mapping")
        return BaselineSpec(label=label, config=dict(cfg))


@dataclass
class HistoryBaselineSpec:
    """
    Baseline(s) derived from a JSON history file.
    """

    file: str
    k: int = 1
    maximize: bool = True
    label_prefix: Optional[str] = None

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "HistoryBaselineSpec":
        file = str(data.get("file", ""))
        if not file:
            raise ValueError("HistoryBaselineSpec requires 'file' field")
        return HistoryBaselineSpec(
            file=file,
            k=int(data.get("k", 1)),
            maximize=bool(data.get("maximize", True)),
            label_prefix=data.get("label_prefix"),
        )


@dataclass
class ValidationSpec:
    """
    Specification for the validation / benchmark suite.
    """

    enabled: bool = False
    name: str = "validation_suite"
    seeds: List[int] = field(default_factory=list)
    budget_per_run: int = 10000
    baselines: List[BaselineSpec] = field(default_factory=list)
    history_baselines: List[HistoryBaselineSpec] = field(default_factory=list)

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ValidationSpec":
        enabled = bool(data.get("enabled", False))
        name = str(data.get("name", "validation_suite"))

        seeds_raw = data.get("seeds", [])
        if not isinstance(seeds_raw, (list, tuple)):
            raise ValueError("'validation.seeds' must be a list")
        seeds = [int(s) for s in seeds_raw]

        budget = int(data.get("budget_per_run", 10000))

        baselines_raw = data.get("baselines", []) or []
        if not isinstance(baselines_raw, (list, tuple)):
            raise ValueError("'validation.baselines' must be a list")
        baselines = [BaselineSpec.from_dict(b) for b in baselines_raw]

        hist_raw = data.get("history_baselines", []) or []
        if not isinstance(hist_raw, (list, tuple)):
            raise ValueError("'validation.history_baselines' must be a list")
        history_baselines = [HistoryBaselineSpec.from_dict(h) for h in hist_raw]

        return ValidationSpec(
            enabled=enabled,
            name=name,
            seeds=seeds,
            budget_per_run=budget,
            baselines=baselines,
            history_baselines=history_baselines,
        )


@dataclass
class ExperimentSpec:
    """
    JSON-backed specification of a tuning + validation experiment.
    """

    name: str
    tuner_kind: TunerKind
    seed: int = 0
    maximize: bool = True
    random: RandomTunerSpec = field(default_factory=RandomTunerSpec)
    racing: Optional[RacingTunerSpec] = None
    sampler: SamplerSpec = field(default_factory=SamplerSpec)
    validation: Optional[ValidationSpec] = None

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ExperimentSpec":
        name = str(data.get("name", "experiment"))
        tuner_kind_str = str(data.get("tuner_kind", "random")).lower()
        if tuner_kind_str not in ("random", "racing"):
            raise ValueError("tuner_kind must be 'random' or 'racing'")

        tuner_kind = TunerKind.RANDOM if tuner_kind_str == "random" else TunerKind.RACING

        seed = int(data.get("seed", 0))
        maximize = bool(data.get("maximize", True))

        random_data = data.get("random", {}) or {}
        racing_data = data.get("racing", None)
        sampler_data = data.get("sampler", {}) or {}
        validation_data = data.get("validation", None)

        random_spec = RandomTunerSpec.from_dict(random_data)
        racing_spec = RacingTunerSpec.from_dict(racing_data) if racing_data is not None else None
        sampler_spec = SamplerSpec.from_dict(sampler_data)
        validation_spec = ValidationSpec.from_dict(validation_data) if validation_data is not None else None

        return ExperimentSpec(
            name=name,
            tuner_kind=tuner_kind,
            seed=seed,
            maximize=maximize,
            random=random_spec,
            racing=racing_spec,
            sampler=sampler_spec,
            validation=validation_spec,
        )


def load_experiment_spec(path: Union[str, Path]) -> ExperimentSpec:
    """
    Load an ExperimentSpec from a JSON file.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, Mapping):
        raise ValueError(f"Experiment config must be a JSON object, got {type(data)}")

    return ExperimentSpec.from_dict(data)


def _build_sampler(spec: SamplerSpec, param_space) -> Sampler:
    """
    Construct a Sampler instance from a SamplerSpec.
    """
    t = spec.type.lower()
    if t == "uniform":
        return UniformSampler(param_space=param_space)
    if t == "model_based":
        return ModelBasedSampler(
            param_space=param_space,
            exploration_prob=spec.exploration_prob,
            min_samples_to_model=spec.min_samples_to_model,
        )
    raise ValueError(f"Unknown sampler.type: {spec.type!r}")


def _build_scenario(params: Mapping[str, Any]) -> Scenario:
    """
    Construct a Scenario instance from a dict of parameters.
    """
    if not isinstance(params, Mapping):
        raise ValueError("Scenario parameters must be a mapping")
    return Scenario(**params)


def _build_validation_suite(
    spec: ValidationSpec,
    task: TuningTask,
) -> BenchmarkSuite:
    """
    Build a BenchmarkSuite from a ValidationSpec and a TuningTask.
    """
    if not spec.seeds:
        raise ValueError("ValidationSpec.seeds must not be empty when validation is enabled")

    instances = list(task.instances)
    if not instances:
        raise ValueError("TuningTask.instances must not be empty for validation")

    return BenchmarkSuite(
        name=spec.name,
        instances=instances,
        seeds=spec.seeds,
        budget_per_run=spec.budget_per_run,
    )


def _build_baselines_from_validation_spec(
    spec: ValidationSpec,
    base_dir: Optional[Path] = None,
) -> List[ConfigSpec]:
    """
    Construct a list of ConfigSpec baselines from a ValidationSpec.
    """
    baselines: List[ConfigSpec] = []

    for b in spec.baselines:
        baselines.append(ConfigSpec(label=b.label, config=dict(b.config)))

    for hb in spec.history_baselines:
        file_path = Path(hb.file)
        if base_dir is not None and not file_path.is_absolute():
            file_path = base_dir / file_path
        specs = load_top_k_as_config_specs(
            path=file_path,
            k=hb.k,
            maximize=hb.maximize,
            label_prefix=hb.label_prefix,
        )
        baselines.extend(specs)

    return baselines


def build_experiment_from_spec(
    spec: ExperimentSpec,
    task: TuningTask,
) -> TuningExperiment:
    """
    Construct a TuningExperiment from an ExperimentSpec and a TuningTask.
    """
    sampler = _build_sampler(spec.sampler, param_space=task.param_space)

    validation_suite = None
    baselines: List[ConfigSpec] = []
    base_dir = Path.cwd()

    if spec.validation is not None and spec.validation.enabled:
        validation_suite = _build_validation_suite(spec.validation, task)
        baselines = _build_baselines_from_validation_spec(spec.validation, base_dir=base_dir)

    if spec.tuner_kind == TunerKind.RANDOM:
        return TuningExperiment(
            name=spec.name,
            task=task,
            tuner_kind=TunerKind.RANDOM,
            max_trials=spec.random.max_trials,
            scenario=None,
            max_initial_configs=0,
            sampler=sampler,
            validation_suite=validation_suite,
            baselines=baselines,
            seed=spec.seed,
            maximize=spec.maximize,
        )

    if spec.tuner_kind == TunerKind.RACING:
        if spec.racing is None:
            raise ValueError("ExperimentSpec.racing must be provided when tuner_kind='racing'")
        scenario = _build_scenario(spec.racing.scenario_params)
        return TuningExperiment(
            name=spec.name,
            task=task,
            tuner_kind=TunerKind.RACING,
            max_trials=0,
            scenario=scenario,
            max_initial_configs=spec.racing.max_initial_configs,
            sampler=sampler,
            validation_suite=validation_suite,
            baselines=baselines,
            seed=spec.seed,
            maximize=spec.maximize,
        )

    raise ValueError(f"Unsupported tuner_kind: {spec.tuner_kind}")


def run_experiment_from_file(
    config_path: Union[str, Path],
    task: TuningTask,
    eval_fn,
    verbose: bool = True,
) -> ExperimentResult:
    """
    High-level helper to load an experiment spec JSON, build, and run it.
    """
    p = Path(config_path)
    spec = load_experiment_spec(p)

    sampler = _build_sampler(spec.sampler, param_space=task.param_space)

    validation_suite = None
    baselines: List[ConfigSpec] = []

    if spec.validation is not None and spec.validation.enabled:
        validation_suite = _build_validation_suite(spec.validation, task)
        baselines = _build_baselines_from_validation_spec(spec.validation, base_dir=p.parent)

    if spec.tuner_kind == TunerKind.RANDOM:
        experiment = TuningExperiment(
            name=spec.name,
            task=task,
            tuner_kind=TunerKind.RANDOM,
            max_trials=spec.random.max_trials,
            scenario=None,
            max_initial_configs=0,
            sampler=sampler,
            validation_suite=validation_suite,
            baselines=baselines,
            seed=spec.seed,
            maximize=spec.maximize,
        )
    elif spec.tuner_kind == TunerKind.RACING:
        if spec.racing is None:
            raise ValueError("ExperimentSpec.racing must be provided when tuner_kind='racing'")
        scenario = _build_scenario(spec.racing.scenario_params)
        experiment = TuningExperiment(
            name=spec.name,
            task=task,
            tuner_kind=TunerKind.RACING,
            max_trials=0,
            scenario=scenario,
            max_initial_configs=spec.racing.max_initial_configs,
            sampler=sampler,
            validation_suite=validation_suite,
            baselines=baselines,
            seed=spec.seed,
            maximize=spec.maximize,
        )
    else:
        raise ValueError(f"Unsupported tuner_kind: {spec.tuner_kind}")

    return experiment.run(eval_fn, verbose=verbose)


def example_spec_usage() -> None:
    """
    Example of how to write, load, and run an experiment spec.
    This is a usage example and should not be executed on import.
    """
    # example_config = {
    #     "name": "zdt1_racing_autonsga",
    #     "tuner_kind": "racing",
    #     "seed": 42,
    #     "maximize": True,
    #     "racing": {
    #         "max_initial_configs": 20,
    #         "scenario": {
    #             "max_experiments": 1000,
    #             "min_survivors": 2,
    #             "elimination_fraction": 0.5,
    #             "instance_order_random": True,
    #             "seed_order_random": True,
    #             "start_instances": 1,
    #             "verbose": True,
    #             "use_statistical_tests": True,
    #             "alpha": 0.05,
    #             "min_blocks_before_elimination": 3,
    #             "use_adaptive_budget": True,
    #             "initial_budget_per_run": 5000,
    #             "max_budget_per_run": 20000,
    #             "budget_growth_factor": 2.0,
    #             "use_elitist_restarts": True,
    #             "target_population_size": 20,
    #             "elite_fraction": 0.3,
    #             "max_elite_archive_size": 20,
    #             "neighbor_fraction": 0.5,
    #         },
    #     },
    #     "sampler": {
    #         "type": "model_based",
    #         "exploration_prob": 0.2,
    #         "min_samples_to_model": 5,
    #     },
    #     "validation": {
    #         "enabled": True,
    #         "name": "ZDT1_suite",
    #         "seeds": [1, 2, 3, 4, 5],
    #         "budget_per_run": 20000,
    #         "baselines": [
    #             {
    #                 "label": "Default NSGA-II",
    #                 "config": {
    #                     "population_size": 100,
    #                     "offspring_size": 100,
    #                     "crossover.type": "sbx",
    #                     "crossover.prob": 0.9,
    #                     "mutation.type": "polynomial",
    #                     "mutation.prob_factor": 1.0,
    #                 },
    #             }
    #         ],
    #         "history_baselines": [
    #             {
    #                 "file": "results/autonsga_phase2_history.json",
    #                 "k": 3,
    #                 "maximize": True,
    #                 "label_prefix": "Racing best",
    #             }
    #         ],
    #     },
    # }
    #
    # with open("experiment_zdt1.json", "w", encoding="utf-8") as fh:
    #     json.dump(example_config, fh, indent=2)
    #
    # # Later:
    # # task = create_zdt1_task()
    # # result = run_experiment_from_file("experiment_zdt1.json", task=task, eval_fn=eval_fn)
    # # print("Best tuned config:", result.best_config)


__all__ = [
    "SamplerSpec",
    "RandomTunerSpec",
    "RacingTunerSpec",
    "BaselineSpec",
    "HistoryBaselineSpec",
    "ValidationSpec",
    "ExperimentSpec",
    "load_experiment_spec",
    "build_experiment_from_spec",
    "run_experiment_from_file",
    "example_spec_usage",
]
