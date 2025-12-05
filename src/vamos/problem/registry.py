from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from .dtlz import DTLZ1Problem, DTLZ2Problem, DTLZ3Problem, DTLZ4Problem
from .binary import (
    BinaryFeatureSelectionProblem,
    BinaryKnapsackProblem,
    BinaryQUBOProblem,
)
from .integer import (
    IntegerJobAssignmentProblem,
    IntegerResourceAllocationProblem,
)
from .mixed import MixedDesignProblem
from .real_world.engineering import WeldedBeamDesignProblem
from .real_world.feature_selection import FeatureSelectionProblem
from .real_world.hyperparam import HyperparameterTuningProblem
from .tsp import TSPProblem
from .lz import (
    LZ09F1Problem,
    LZ09F2Problem,
    LZ09F3Problem,
    LZ09F4Problem,
    LZ09F5Problem,
    LZ09F6Problem,
    LZ09F7Problem,
    LZ09F8Problem,
    LZ09F9Problem,
)
from .cec import (
    CEC2009CF1Problem,
    CEC2009UF1Problem,
    CEC2009UF2Problem,
    CEC2009UF3Problem,
)
from .wfg import (
    WFG1Problem,
    WFG2Problem,
    WFG3Problem,
    WFG4Problem,
    WFG5Problem,
    WFG6Problem,
    WFG7Problem,
    WFG8Problem,
    WFG9Problem,
)
from .zdt1 import ZDT1Problem
from .zdt2 import ZDT2Problem
from .zdt3 import ZDT3Problem
from .zdt4 import ZDT4Problem
from .zdt6 import ZDT6Problem

ProblemFactory = Callable[[int, Optional[int]], object]


@dataclass(frozen=True)
class ProblemSpec:
    """Metadata and factory for a benchmark problem."""
    key: str
    label: str
    default_n_var: int
    default_n_obj: int
    allow_n_obj_override: bool
    factory: ProblemFactory
    description: str = ""
    encoding: str = "continuous"

    def resolve_dimensions(
        self, *, n_var: Optional[int], n_obj: Optional[int]
    ) -> Tuple[int, int]:
        """
        Apply default dimensions and enforce override rules.
        """
        actual_n_var = n_var if n_var is not None else self.default_n_var
        if actual_n_var <= 0:
            raise ValueError("n_var must be a positive integer.")

        if self.allow_n_obj_override:
            actual_n_obj = n_obj if n_obj is not None else self.default_n_obj
            if actual_n_obj <= 0:
                raise ValueError("n_obj must be a positive integer.")
        else:
            actual_n_obj = self.default_n_obj
            if n_obj is not None and n_obj != actual_n_obj:
                raise ValueError(
                    f"Problem '{self.label}' has a fixed number of objectives "
                    f"({self.default_n_obj}). --n-obj overrides are not supported."
                )

        return actual_n_var, actual_n_obj


@dataclass(frozen=True)
class ProblemSelection:
    """Concrete problem instance choice with resolved dimensions."""
    spec: ProblemSpec
    n_var: int
    n_obj: int

    def instantiate(self):
        """
        Create a new problem instance with the resolved dimensions.
        """
        return self.spec.factory(self.n_var, self.n_obj)


def _zdt1_factory(n_var: int, _ignored: Optional[int] = None):
    return ZDT1Problem(n_var=n_var)


def _dtlz_factory(cls, n_var: int, n_obj: Optional[int]):
    return cls(n_var=n_var, n_obj=n_obj if n_obj is not None else 3)


def _tsp_factory(n_var: int, _ignored: Optional[int] = None):
    return TSPProblem(n_cities=n_var)


def _tsplib_tsp_factory(dataset: str):
    def _factory(_n_var: int, _ignored: Optional[int] = None):
        return TSPProblem(dataset=dataset)

    return _factory


PROBLEM_SPECS: Dict[str, ProblemSpec] = {
    "zdt1": ProblemSpec(
        key="zdt1",
        label="ZDT1",
        default_n_var=30,
        default_n_obj=2,
        allow_n_obj_override=False,
        description="Classic bi-objective benchmark with a convex Pareto front.",
        factory=_zdt1_factory,
    ),
    "zdt2": ProblemSpec(
        key="zdt2",
        label="ZDT2",
        default_n_var=30,
        default_n_obj=2,
        allow_n_obj_override=False,
        description="ZDT variant with a concave Pareto front.",
        factory=lambda n_var, _n_obj: ZDT2Problem(n_var=n_var),
    ),
    "zdt3": ProblemSpec(
        key="zdt3",
        label="ZDT3",
        default_n_var=30,
        default_n_obj=2,
        allow_n_obj_override=False,
        description="ZDT benchmark with a disconnected Pareto front.",
        factory=lambda n_var, _n_obj: ZDT3Problem(n_var=n_var),
    ),
    "zdt4": ProblemSpec(
        key="zdt4",
        label="ZDT4",
        default_n_var=10,
        default_n_obj=2,
        allow_n_obj_override=False,
        description="ZDT benchmark with multimodal landscape and mixed bounds.",
        factory=lambda n_var, _n_obj: ZDT4Problem(n_var=n_var),
    ),
    "zdt6": ProblemSpec(
        key="zdt6",
        label="ZDT6",
        default_n_var=10,
        default_n_obj=2,
        allow_n_obj_override=False,
        description="ZDT benchmark with non-uniform density over the Pareto set.",
        factory=lambda n_var, _n_obj: ZDT6Problem(n_var=n_var),
    ),
    "dtlz1": ProblemSpec(
        key="dtlz1",
        label="DTLZ1",
        default_n_var=7,
        default_n_obj=3,
        allow_n_obj_override=True,
        description="DTLZ1 with configurable objectives (defaults to 3).",
        factory=lambda n_var, n_obj: _dtlz_factory(DTLZ1Problem, n_var, n_obj),
    ),
    "dtlz2": ProblemSpec(
        key="dtlz2",
        label="DTLZ2",
        default_n_var=12,
        default_n_obj=3,
        allow_n_obj_override=True,
        description="DTLZ2 with configurable objectives (defaults to 3).",
        factory=lambda n_var, n_obj: _dtlz_factory(DTLZ2Problem, n_var, n_obj),
    ),
    "dtlz3": ProblemSpec(
        key="dtlz3",
        label="DTLZ3",
        default_n_var=12,
        default_n_obj=3,
        allow_n_obj_override=True,
        description="DTLZ3 with configurable objectives (defaults to 3).",
        factory=lambda n_var, n_obj: _dtlz_factory(DTLZ3Problem, n_var, n_obj),
    ),
    "dtlz4": ProblemSpec(
        key="dtlz4",
        label="DTLZ4",
        default_n_var=12,
        default_n_obj=3,
        allow_n_obj_override=True,
        description="DTLZ4 with configurable objectives (defaults to 3).",
        factory=lambda n_var, n_obj: _dtlz_factory(DTLZ4Problem, n_var, n_obj),
    ),
    "lz09_f1": ProblemSpec(
        key="lz09_f1",
        label="LZ09 F1",
        default_n_var=10,
        default_n_obj=2,
        allow_n_obj_override=False,
        description="Li & Zhang LZ09 F1 benchmark.",
        factory=lambda n_var, _n_obj: LZ09F1Problem(n_var=n_var),
    ),
    "lz09_f2": ProblemSpec(
        key="lz09_f2",
        label="LZ09 F2",
        default_n_var=30,
        default_n_obj=2,
        allow_n_obj_override=False,
        description="Li & Zhang LZ09 F2 benchmark.",
        factory=lambda n_var, _n_obj: LZ09F2Problem(n_var=n_var),
    ),
    "lz09_f3": ProblemSpec(
        key="lz09_f3",
        label="LZ09 F3",
        default_n_var=30,
        default_n_obj=2,
        allow_n_obj_override=False,
        description="Li & Zhang LZ09 F3 benchmark.",
        factory=lambda n_var, _n_obj: LZ09F3Problem(n_var=n_var),
    ),
    "lz09_f4": ProblemSpec(
        key="lz09_f4",
        label="LZ09 F4",
        default_n_var=30,
        default_n_obj=2,
        allow_n_obj_override=False,
        description="Li & Zhang LZ09 F4 benchmark.",
        factory=lambda n_var, _n_obj: LZ09F4Problem(n_var=n_var),
    ),
    "lz09_f5": ProblemSpec(
        key="lz09_f5",
        label="LZ09 F5",
        default_n_var=30,
        default_n_obj=2,
        allow_n_obj_override=False,
        description="Li & Zhang LZ09 F5 benchmark.",
        factory=lambda n_var, _n_obj: LZ09F5Problem(n_var=n_var),
    ),
    "lz09_f6": ProblemSpec(
        key="lz09_f6",
        label="LZ09 F6",
        default_n_var=10,
        default_n_obj=3,
        allow_n_obj_override=False,
        description="Li & Zhang LZ09 F6 benchmark (tri-objective).",
        factory=lambda n_var, _n_obj: LZ09F6Problem(n_var=n_var),
    ),
    "lz09_f7": ProblemSpec(
        key="lz09_f7",
        label="LZ09 F7",
        default_n_var=10,
        default_n_obj=2,
        allow_n_obj_override=False,
        description="Li & Zhang LZ09 F7 benchmark.",
        factory=lambda n_var, _n_obj: LZ09F7Problem(n_var=n_var),
    ),
    "lz09_f8": ProblemSpec(
        key="lz09_f8",
        label="LZ09 F8",
        default_n_var=10,
        default_n_obj=2,
        allow_n_obj_override=False,
        description="Li & Zhang LZ09 F8 benchmark.",
        factory=lambda n_var, _n_obj: LZ09F8Problem(n_var=n_var),
    ),
    "lz09_f9": ProblemSpec(
        key="lz09_f9",
        label="LZ09 F9",
        default_n_var=30,
        default_n_obj=2,
        allow_n_obj_override=False,
        description="Li & Zhang LZ09 F9 benchmark.",
        factory=lambda n_var, _n_obj: LZ09F9Problem(n_var=n_var),
    ),
    "cec2009_uf1": ProblemSpec(
        key="cec2009_uf1",
        label="CEC2009 UF1",
        default_n_var=30,
        default_n_obj=2,
        allow_n_obj_override=False,
        description="CEC2009 unconstrained function UF1.",
        factory=lambda n_var, _n_obj: CEC2009UF1Problem(n_var=n_var),
    ),
    "cec2009_uf2": ProblemSpec(
        key="cec2009_uf2",
        label="CEC2009 UF2",
        default_n_var=30,
        default_n_obj=2,
        allow_n_obj_override=False,
        description="CEC2009 unconstrained function UF2.",
        factory=lambda n_var, _n_obj: CEC2009UF2Problem(n_var=n_var),
    ),
    "cec2009_uf3": ProblemSpec(
        key="cec2009_uf3",
        label="CEC2009 UF3",
        default_n_var=30,
        default_n_obj=2,
        allow_n_obj_override=False,
        description="CEC2009 unconstrained function UF3.",
        factory=lambda n_var, _n_obj: CEC2009UF3Problem(n_var=n_var),
    ),
    "cec2009_cf1": ProblemSpec(
        key="cec2009_cf1",
        label="CEC2009 CF1",
        default_n_var=30,
        default_n_obj=2,
        allow_n_obj_override=False,
        description="CEC2009 constrained function CF1.",
        factory=lambda n_var, _n_obj: CEC2009CF1Problem(n_var=n_var),
    ),
    "wfg1": ProblemSpec(
        key="wfg1",
        label="WFG1",
        default_n_var=24,
        default_n_obj=3,
        allow_n_obj_override=True,
        description="WFG1 benchmarking problem (requires pymoo).",
        factory=lambda n_var, n_obj: WFG1Problem(
            n_var=n_var,
            n_obj=n_obj if n_obj is not None else 3,
        ),
    ),
    "wfg2": ProblemSpec(
        key="wfg2",
        label="WFG2",
        default_n_var=24,
        default_n_obj=3,
        allow_n_obj_override=True,
        description="WFG2 benchmarking problem (requires pymoo).",
        factory=lambda n_var, n_obj: WFG2Problem(
            n_var=n_var,
            n_obj=n_obj if n_obj is not None else 3,
        ),
    ),
    "wfg3": ProblemSpec(
        key="wfg3",
        label="WFG3",
        default_n_var=24,
        default_n_obj=3,
        allow_n_obj_override=True,
        description="WFG3 benchmarking problem (requires pymoo).",
        factory=lambda n_var, n_obj: WFG3Problem(
            n_var=n_var,
            n_obj=n_obj if n_obj is not None else 3,
        ),
    ),
    "wfg4": ProblemSpec(
        key="wfg4",
        label="WFG4",
        default_n_var=24,
        default_n_obj=3,
        allow_n_obj_override=True,
        description="WFG4 benchmarking problem (requires pymoo).",
        factory=lambda n_var, n_obj: WFG4Problem(
            n_var=n_var,
            n_obj=n_obj if n_obj is not None else 3,
        ),
    ),
    "wfg5": ProblemSpec(
        key="wfg5",
        label="WFG5",
        default_n_var=24,
        default_n_obj=3,
        allow_n_obj_override=True,
        description="WFG5 benchmarking problem (requires pymoo).",
        factory=lambda n_var, n_obj: WFG5Problem(
            n_var=n_var,
            n_obj=n_obj if n_obj is not None else 3,
        ),
    ),
    "wfg6": ProblemSpec(
        key="wfg6",
        label="WFG6",
        default_n_var=24,
        default_n_obj=3,
        allow_n_obj_override=True,
        description="WFG6 benchmarking problem (requires pymoo).",
        factory=lambda n_var, n_obj: WFG6Problem(
            n_var=n_var,
            n_obj=n_obj if n_obj is not None else 3,
        ),
    ),
    "wfg7": ProblemSpec(
        key="wfg7",
        label="WFG7",
        default_n_var=24,
        default_n_obj=3,
        allow_n_obj_override=True,
        description="WFG7 benchmarking problem (requires pymoo).",
        factory=lambda n_var, n_obj: WFG7Problem(
            n_var=n_var,
            n_obj=n_obj if n_obj is not None else 3,
        ),
    ),
    "wfg8": ProblemSpec(
        key="wfg8",
        label="WFG8",
        default_n_var=24,
        default_n_obj=3,
        allow_n_obj_override=True,
        description="WFG8 benchmarking problem (requires pymoo).",
        factory=lambda n_var, n_obj: WFG8Problem(
            n_var=n_var,
            n_obj=n_obj if n_obj is not None else 3,
        ),
    ),
    "wfg9": ProblemSpec(
        key="wfg9",
        label="WFG9",
        default_n_var=24,
        default_n_obj=3,
        allow_n_obj_override=True,
        description="WFG9 benchmarking problem (requires pymoo).",
        factory=lambda n_var, n_obj: WFG9Problem(
            n_var=n_var,
            n_obj=n_obj if n_obj is not None else 3,
        ),
    ),
    "tsp6": ProblemSpec(
        key="tsp6",
        label="TSP (6 cities)",
        default_n_var=6,
        default_n_obj=2,
        allow_n_obj_override=False,
        encoding="permutation",
        description="Toy traveling salesman instance with 6 cities (permutation encoding).",
        factory=_tsp_factory,
    ),
    "kroa100": ProblemSpec(
        key="kroa100",
        label="TSPLIB KroA100",
        default_n_var=100,
        default_n_obj=2,
        allow_n_obj_override=False,
        encoding="permutation",
        description="TSPLIB 100-city KroA instance (permutation encoding).",
        factory=_tsplib_tsp_factory("kroA100"),
    ),
    "krob100": ProblemSpec(
        key="krob100",
        label="TSPLIB KroB100",
        default_n_var=100,
        default_n_obj=2,
        allow_n_obj_override=False,
        encoding="permutation",
        description="TSPLIB 100-city KroB instance (permutation encoding).",
        factory=_tsplib_tsp_factory("kroB100"),
    ),
    "kroc100": ProblemSpec(
        key="kroc100",
        label="TSPLIB KroC100",
        default_n_var=100,
        default_n_obj=2,
        allow_n_obj_override=False,
        encoding="permutation",
        description="TSPLIB 100-city KroC instance (permutation encoding).",
        factory=_tsplib_tsp_factory("kroC100"),
    ),
    "krod100": ProblemSpec(
        key="krod100",
        label="TSPLIB KroD100",
        default_n_var=100,
        default_n_obj=2,
        allow_n_obj_override=False,
        encoding="permutation",
        description="TSPLIB 100-city KroD instance (permutation encoding).",
        factory=_tsplib_tsp_factory("kroD100"),
    ),
    "kroe100": ProblemSpec(
        key="kroe100",
        label="TSPLIB KroE100",
        default_n_var=100,
        default_n_obj=2,
        allow_n_obj_override=False,
        encoding="permutation",
        description="TSPLIB 100-city KroE instance (permutation encoding).",
        factory=_tsplib_tsp_factory("kroE100"),
    ),
    "bin_feat": ProblemSpec(
        key="bin_feat",
        label="Binary Feature Selection",
        default_n_var=50,
        default_n_obj=2,
        allow_n_obj_override=False,
        encoding="binary",
        description="Synthetic feature-selection style binary benchmark.",
        factory=lambda n_var, _n_obj: BinaryFeatureSelectionProblem(n_var=n_var),
    ),
    "bin_knapsack": ProblemSpec(
        key="bin_knapsack",
        label="Binary Knapsack",
        default_n_var=50,
        default_n_obj=2,
        allow_n_obj_override=False,
        encoding="binary",
        description="Knapsack-like binary benchmark trading value vs capacity deviation.",
        factory=lambda n_var, _n_obj: BinaryKnapsackProblem(n_var=n_var),
    ),
    "bin_qubo": ProblemSpec(
        key="bin_qubo",
        label="Binary QUBO",
        default_n_var=30,
        default_n_obj=2,
        allow_n_obj_override=False,
        encoding="binary",
        description="Quadratic unconstrained binary optimization surrogate.",
        factory=lambda n_var, _n_obj: BinaryQUBOProblem(n_var=n_var),
    ),
    "int_alloc": ProblemSpec(
        key="int_alloc",
        label="Integer Resource Allocation",
        default_n_var=20,
        default_n_obj=2,
        allow_n_obj_override=False,
        encoding="integer",
        description="Resource allocation with integer budgets and diminishing returns.",
        factory=lambda n_var, _n_obj: IntegerResourceAllocationProblem(n_var=n_var),
    ),
    "int_jobs": ProblemSpec(
        key="int_jobs",
        label="Integer Job Assignment",
        default_n_var=30,
        default_n_obj=2,
        allow_n_obj_override=False,
        encoding="integer",
        description="Assign job types to positions with mismatch and diversity objectives.",
        factory=lambda n_var, _n_obj: IntegerJobAssignmentProblem(n_positions=n_var),
    ),
    "mixed_design": ProblemSpec(
        key="mixed_design",
        label="Mixed Design",
        default_n_var=9,
        default_n_obj=2,
        allow_n_obj_override=False,
        encoding="mixed",
        description="Mixed real/integer/categorical benchmark.",
        factory=lambda n_var, _n_obj: MixedDesignProblem(n_var=n_var),
    ),
    "ml_tuning": ProblemSpec(
        key="ml_tuning",
        label="ML Hyperparameter Tuning",
        default_n_var=4,
        default_n_obj=2,
        allow_n_obj_override=False,
        encoding="mixed",
        description="SVM hyperparameter tuning on a small dataset (requires scikit-learn).",
        factory=lambda _n_var, _n_obj: HyperparameterTuningProblem(),
    ),
    "welded_beam": ProblemSpec(
        key="welded_beam",
        label="Welded Beam Design",
        default_n_var=6,
        default_n_obj=2,
        allow_n_obj_override=False,
        encoding="mixed",
        description="Classic welded-beam design with mixed variables and constraints.",
        factory=lambda _n_var, _n_obj: WeldedBeamDesignProblem(),
    ),
    "fs_real": ProblemSpec(
        key="fs_real",
        label="Feature Selection (real data)",
        default_n_var=30,
        default_n_obj=2,
        allow_n_obj_override=False,
        encoding="binary",
        description="Binary feature selection on a real dataset (requires scikit-learn).",
        factory=lambda _n_var, _n_obj: FeatureSelectionProblem(),
    ),
}


def available_problem_names() -> Tuple[str, ...]:
    return tuple(PROBLEM_SPECS.keys())


def make_problem_selection(
    key: str, *, n_var: Optional[int] = None, n_obj: Optional[int] = None
) -> ProblemSelection:
    try:
        spec = PROBLEM_SPECS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown problem '{key}'.") from exc

    actual_n_var, actual_n_obj = spec.resolve_dimensions(n_var=n_var, n_obj=n_obj)
    return ProblemSelection(spec=spec, n_var=actual_n_var, n_obj=actual_n_obj)
