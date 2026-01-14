from __future__ import annotations

from dataclasses import dataclass

from vamos.engine.config.spec import ExperimentSpec, ProblemOverrides, SpecBlock


@dataclass(frozen=True)
class SpecDefaults:
    spec: ExperimentSpec
    problem_overrides: ProblemOverrides
    experiment_defaults: SpecBlock
    nsgaii_defaults: SpecBlock
    moead_defaults: SpecBlock
    smsemoa_defaults: SpecBlock
    nsgaiii_defaults: SpecBlock
