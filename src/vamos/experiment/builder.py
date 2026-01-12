from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing_extensions import Self

from vamos.experiment.optimize import OptimizeConfig, OptimizationResult, optimize_config
from vamos.foundation.eval.backends import EvaluationBackend, resolve_eval_strategy
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.problem.types import ProblemProtocol
from vamos.exceptions import InvalidAlgorithmError


@dataclass(frozen=True)
class _DictAlgorithmConfig:
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


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
        self._live_viz: Any | None = None
        self._eval_strategy: str | EvaluationBackend | None = None
        self._eval_n_workers: int | None = None
        self._eval_chunk_size: int | None = None
        self._eval_dask_address: str | None = None

    def using(self, algorithm: str, **kwargs: Any) -> "Self":
        """
        Select the algorithm to use.

        Args:
            algorithm: Algorithm name (e.g., 'nsgaii', 'moead')
            **kwargs: Algorithm-specific parameters (e.g., pop_size)
        """
        self._algorithm = algorithm
        pop_size = kwargs.pop("pop_size", None)
        if pop_size is not None:
            self._pop_size = int(pop_size)
        self._algo_kwargs.update(kwargs)
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

    def eval_strategy(
        self,
        strategy: str | EvaluationBackend,
        *,
        n_workers: int | None = None,
        chunk_size: int | None = None,
        dask_address: str | None = None,
    ) -> "Self":
        """
        Set the evaluation strategy for objective evaluations.

        Args:
            strategy: Strategy name ("serial", "multiprocessing", "dask") or a concrete EvaluationBackend instance.
            n_workers: Worker count for multiprocessing.
            chunk_size: Chunk size for multiprocessing.
            dask_address: Scheduler address for the dask backend.
        """
        self._eval_strategy = strategy
        self._eval_n_workers = n_workers
        self._eval_chunk_size = chunk_size
        self._eval_dask_address = dask_address
        return self

    def run(self) -> OptimizationResult:
        """
        Execute the optimization study.

        Returns:
            OptimizationResult containing the Pareto front and history.
        """
        problem = self._resolve_problem()
        pop_size = self._pop_size if self._pop_size is not None else 100
        seed = self._seed if self._seed is not None else 42

        algo_kwargs = dict(self._algo_kwargs)
        algo_cfg = self._resolve_algorithm_config(
            problem,
            pop_size=pop_size,
            engine=self._engine,
            algo_kwargs=algo_kwargs,
        )

        eval_backend = self._resolve_eval_backend()
        config = OptimizeConfig(
            problem=problem,
            algorithm=self._algorithm,
            algorithm_config=algo_cfg,
            termination=("n_eval", self._max_evaluations),
            seed=seed,
            engine=self._engine,
            eval_strategy=eval_backend,
            live_viz=self._live_viz,
        )
        return optimize_config(config)

    def _resolve_problem(self) -> ProblemProtocol:
        if isinstance(self._problem, str):
            selection = make_problem_selection(
                self._problem,
                n_var=self._problem_kwargs.get("n_var"),
                n_obj=self._problem_kwargs.get("n_obj"),
            )
            return selection.instantiate()
        if isinstance(self._problem, type):
            return cast(type[ProblemProtocol], self._problem)()
        return cast(ProblemProtocol, self._problem)

    def _resolve_eval_backend(self) -> EvaluationBackend | None:
        if self._eval_strategy is None:
            return None
        if isinstance(self._eval_strategy, str):
            return resolve_eval_strategy(
                self._eval_strategy,
                n_workers=self._eval_n_workers,
                chunk_size=self._eval_chunk_size,
                dask_address=self._eval_dask_address,
            )
        return self._eval_strategy

    def _resolve_algorithm_config(
        self,
        problem: ProblemProtocol,
        *,
        pop_size: int,
        engine: str,
        algo_kwargs: dict[str, Any],
    ) -> Any:
        from vamos.experiment.optimize import _build_algorithm_config

        n_var = getattr(problem, "n_var", None)
        n_obj = getattr(problem, "n_obj", None)

        try:
            return _build_algorithm_config(
                self._algorithm,
                pop_size=pop_size,
                n_var=n_var,
                n_obj=n_obj,
                engine=engine,
                **algo_kwargs,
            )
        except InvalidAlgorithmError:
            cfg_data = dict(algo_kwargs)
            cfg_data["engine"] = engine
            cfg_data["pop_size"] = pop_size
            return _DictAlgorithmConfig(cfg_data)


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
