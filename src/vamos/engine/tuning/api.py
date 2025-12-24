from __future__ import annotations

"""
Facade for tuning utilities (racing-style tuning).

Parameter types use signature: ParamType(name, ...)
- Real(name, low, high, log=False)
- Int(name, low, high, log=False)
- Categorical(name, choices)
- Boolean(name)

For full control, import from `vamos.engine.tuning.*`.
"""

from vamos.engine.tuning import (
    # Tuners
    AlgorithmConfigSpace,
    RandomSearchTuner,
    RacingTuner,
    # Parameter types
    ParamSpace,
    Real,
    Int,
    Categorical,
    Boolean,
    Condition,
    ConditionalBlock,
    # Task/scenario
    TuningTask,
    EvalContext,
    Instance,
    Scenario,
    # Config space builders
    build_ibea_config_space,
    build_moead_config_space,
    build_nsgaiii_config_space,
    build_nsgaii_config_space,
    build_smpso_config_space,
    build_smsemoa_config_space,
    build_spea2_config_space,
    config_from_assignment,
)

__all__ = [
    # Tuners
    "AlgorithmConfigSpace",
    "RandomSearchTuner",
    "RacingTuner",
    # Parameter types
    "ParamSpace",
    "Real",
    "Int",
    "Categorical",
    "Boolean",
    "Condition",
    "ConditionalBlock",
    # Task/scenario
    "TuningTask",
    "EvalContext",
    "Instance",
    "Scenario",
    # Config space builders
    "build_nsgaii_config_space",
    "build_nsgaiii_config_space",
    "build_moead_config_space",
    "build_smsemoa_config_space",
    "build_spea2_config_space",
    "build_ibea_config_space",
    "build_smpso_config_space",
    "config_from_assignment",
]
