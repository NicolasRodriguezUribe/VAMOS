from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

from vamos.experiment.optimize import OptimizationResult
from vamos.experiment.runner import run_single
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import make_problem_selection


class StudyBuilder:
    """
    Fluent builder for configuring and running optimization studies.

    Example:
        >>> result = vamos.study("zdt1", n_var=30) \
        ...     .using("nsgaii", pop_size=100) \
        ...     .engine("numba") \
        ...     .evaluations(5000) \
        ...     .seed(42) \
        ...     .run()
    """

    def __init__(self, problem: Any | str, **problem_kwargs: Any):
        """
        Initialize the builder with a problem.

        Args:
            problem: Problem name (str) or instance
            **problem_kwargs: Arguments for problem instantiation (e.g., n_var)
        """
        self._problem = problem
        self._problem_kwargs = problem_kwargs

        # Defaults
        self._algorithm = "nsgaii"
        self._algo_kwargs: dict[str, Any] = {}
        self._engine = "numpy"
        self._seed: Optional[int] = None
        self._max_evaluations = 10000
        self._pop_size: Optional[int] = None

        # Advanced config
        self._live_viz = False
        self._eval_strategy = "serial"

    def using(self, algorithm: str, **kwargs: Any) -> "Self":
        """
        Select the algorithm to use.

        Args:
            algorithm: Algorithm name (e.g., 'nsgaii', 'moead')
            **kwargs: Algorithm-specific parameters (e.g., pop_size)
        """
        self._algorithm = algorithm
        self._algo_kwargs.update(kwargs)
        if "pop_size" in kwargs:
            self._pop_size = kwargs["pop_size"]
        return self

    def engine(self, engine: str) -> "Self":
        """
        Set the execution engine.

        Args:
            engine: 'numpy' (default), 'numba', or 'moocore'
        """
        self._engine = engine
        return self

    def evaluations(self, n_evals: int) -> "Self":
        """Set the maximum number of function evaluations."""
        self._max_evaluations = n_evals
        return self

    def seed(self, seed: int) -> "Self":
        """Set the random seed for reproducibility."""
        self._seed = seed
        return self

    def run(self) -> OptimizationResult:
        """
        Execute the optimization study.

        Returns:
            OptimizationResult containing the Pareto front and history.
        """
        # 1. Resolve Problem
        if isinstance(self._problem, str):
            selection = make_problem_selection(
                self._problem, n_var=self._problem_kwargs.get("n_var"), n_obj=self._problem_kwargs.get("n_obj")
            )
        else:
            # Wrap existing instance into a dummy selection/spec?
            # Wiring logic expects ProblemSelection.
            # Currently make_problem_selection only handles string keys.
            # If user passes an instance, we might need a simpler path or an adapter.
            # For Phase 3.1, let's support string keys primarily, similar to existing APIs.
            # If instance support is needed, we need to bypass 'make_problem_selection'.

            # WORKAROUND: If instance, we need to construct a Selection that yields this instance.
            # But 'selection.instantiate()' is called inside run_single.

            # Let's support string keys first as per plan.
            raise NotImplementedError("Passing problem instance directly is not yet supported. Use problem name string.")

        # 2. Build Config
        config = ExperimentConfig(
            population_size=self._pop_size if self._pop_size else 100,
            max_evaluations=self._max_evaluations,
            seed=self._seed if self._seed is not None else 42,
            eval_strategy=self._eval_strategy,
            live_viz=self._live_viz,
        )

        # 3. Parameter Mapping (algo kwargs -> config fields or variations)
        # run_single expects specific args like 'selection_pressure', 'nsgaii_variation', etc.
        # We need to map kwargs to these specific arguments.

        run_kwargs = {}

        # Generic args
        if "selection_pressure" in self._algo_kwargs:
            run_kwargs["selection_pressure"] = self._algo_kwargs["selection_pressure"]

        # Variation overrides
        # Assuming kwarg key matches config key (e.g. nsgaii_variation)
        # OR we construct it dynamically.
        # For simplicity, we can pass algo kwargs deeply into the VariationConfig if we knew which one.
        # But 'run_single' takes specific arguments.

        # Let's map 'pop_size' which is common.
        # Already handled via config.population_size.

        # For now, simplistic mapping.

        result_dict = run_single(engine_name=self._engine, algorithm_name=self._algorithm, selection=selection, config=config, **run_kwargs)

        return OptimizationResult(result_dict)


def study(problem: Any | str, **kwargs: Any) -> StudyBuilder:
    """
    Create a new optimization study.

    Args:
        problem: Problem name (e.g., "zdt1")
        **kwargs: Problem arguments (e.g., n_var=30)

    Returns:
        StudyBuilder instance
    """
    return StudyBuilder(problem, **kwargs)
