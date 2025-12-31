from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Literal, Mapping, Tuple, overload

import numpy as np

from vamos.engine.algorithm.registry import resolve_algorithm, ALGORITHMS
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.foundation.problem.types import ProblemProtocol
from vamos.foundation.eval.backends import resolve_eval_backend, EvaluationBackend


@overload
def pareto_filter(
    F: np.ndarray | None, *, return_indices: Literal[False] = False
) -> np.ndarray | None:
    ...


@overload
def pareto_filter(
    F: np.ndarray | None, *, return_indices: Literal[True]
) -> tuple[np.ndarray, np.ndarray]:
    ...


def pareto_filter(
    F: np.ndarray | None, *, return_indices: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
    """
    Return the non-dominated subset of points (first Pareto front).

    Args:
        F: Objective values array (n_solutions, n_objectives) or None.
        return_indices: When True, also return indices of the front in F.

    Returns:
        Front array, or (front, indices) when return_indices is True.
    """
    if F is None:
        if return_indices:
            return np.empty((0, 0)), np.array([], dtype=int)
        return None
    F = np.asarray(F)
    if F.size == 0 or F.ndim < 2:
        if return_indices:
            n = int(F.shape[0]) if F.ndim > 0 else 0
            idx = np.arange(n, dtype=int)
            return F, idx
        return F
    from vamos.foundation.kernel.numpy_backend import _fast_non_dominated_sort

    fronts, _ = _fast_non_dominated_sort(F)
    if not fronts or not fronts[0]:
        if return_indices:
            idx = np.arange(int(F.shape[0]), dtype=int)
            return F, idx
        return F
    idx = np.asarray(fronts[0], dtype=int)
    front = F[idx]
    return (front, idx) if return_indices else front


class OptimizationResult:
    """
    Container returned by optimize() with user-friendly helper methods.

    Attributes:
        F: Objective values array (n_solutions, n_objectives)
        X: Decision variables array (n_solutions, n_variables), may be None
        data: Full result dictionary with all fields

    Examples:
        >>> result = optimize(config)
        >>> result.summary()  # Print quick overview
        >>> result.plot()  # Visualize Pareto front
        >>> best = result.best("knee")  # Select a solution
        >>> df = result.to_dataframe()  # Export to pandas
    """

    def __init__(self, payload: Mapping[str, Any]):
        self.F = payload.get("F")
        self.X = payload.get("X")
        self.data = dict(payload)

    def __len__(self) -> int:
        """Number of solutions in the result."""
        return len(self.F) if self.F is not None else 0

    def __repr__(self) -> str:
        n_sol = len(self)
        n_obj = self.F.shape[1] if self.F is not None and len(self.F) > 0 else 0
        return f"OptimizationResult({n_sol} solutions, {n_obj} objectives)"

    @property
    def n_objectives(self) -> int:
        """Number of objectives."""
        return self.F.shape[1] if self.F is not None and len(self.F) > 0 else 0

    def summary(self) -> None:
        """Print a summary of the optimization result."""
        print("=== Optimization Result ===")
        print(f"Solutions: {len(self)}")
        print(f"Objectives: {self.n_objectives}")

        if self.F is not None and len(self.F) > 0:
            print("\nObjective ranges:")
            for i in range(self.n_objectives):
                print(f"  f{i+1}: [{self.F[:, i].min():.6f}, {self.F[:, i].max():.6f}]")

    @overload
    def front(
        self, *, return_indices: Literal[False] = False
    ) -> np.ndarray | None:
        ...

    @overload
    def front(
        self, *, return_indices: Literal[True]
    ) -> tuple[np.ndarray, np.ndarray]:
        ...

    def front(
        self, *, return_indices: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
        """
        Return non-dominated solutions (first Pareto front).

        Args:
            return_indices: When True, also return indices of the front in F.
        """
        return pareto_filter(self.F, return_indices=return_indices)

    def plot(self, show: bool = True, **kwargs: Any) -> Any:
        """
        Plot the Pareto front (2D or 3D).

        Args:
            show: Whether to display the plot immediately
            **kwargs: Additional arguments passed to scatter plot

        Returns:
            matplotlib Axes object

        Raises:
            ImportError: If matplotlib is not installed
            ValueError: If more than 3 objectives
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            ) from exc

        F_plot = self.front()
        if F_plot is None or len(F_plot) == 0:
            raise ValueError("No solutions to plot")

        n_obj = self.n_objectives
        if n_obj == 2:
            fig, ax = plt.subplots()
            ax.scatter(F_plot[:, 0], F_plot[:, 1], **kwargs)
            ax.set_xlabel("f1")
            ax.set_ylabel("f2")
            ax.set_title("Pareto Front")
        elif n_obj == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(F_plot[:, 0], F_plot[:, 1], F_plot[:, 2], **kwargs)
            ax.set_xlabel("f1")
            ax.set_ylabel("f2")
            ax.set_zlabel("f3")
            ax.set_title("Pareto Front")
        else:
            raise ValueError(
                f"Cannot plot {n_obj} objectives. Use to_dataframe() for analysis."
            )

        if show:
            plt.show()
        return ax

    def best(self, method: str = "knee") -> dict[str, Any]:
        """
        Select a single 'best' solution from the Pareto front.

        Args:
            method: Selection method - 'knee' (default), 'min_f1', 'min_f2', 'balanced'

        Returns:
            Dictionary with 'X' (decision vars), 'F' (objectives),
            'index' (position in front)
        """
        if self.F is None or len(self.F) == 0:
            raise ValueError("No solutions available")

        if method == "knee":
            # Simple knee point: minimize normalized L1 distance
            F_norm = (self.F - self.F.min(axis=0)) / (np.ptp(self.F, axis=0) + 1e-12)
            idx = int(np.argmin(F_norm.sum(axis=1)))
        elif method == "min_f1":
            idx = int(np.argmin(self.F[:, 0]))
        elif method == "min_f2":
            idx = int(np.argmin(self.F[:, 1]))
        elif method == "balanced":
            # Minimize max normalized objective
            F_norm = (self.F - self.F.min(axis=0)) / (np.ptp(self.F, axis=0) + 1e-12)
            idx = int(np.argmin(F_norm.max(axis=1)))
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use: knee, min_f1, min_f2, balanced"
            )

        return {
            "X": self.X[idx] if self.X is not None else None,
            "F": self.F[idx],
            "index": idx,
        }

    def to_dataframe(self) -> Any:
        """
        Convert results to a pandas DataFrame.

        Returns:
            DataFrame with columns for each objective (f1, f2, ...) and optionally
            decision variables (x1, x2, ...).

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            ) from exc

        if self.F is None or len(self.F) == 0:
            return pd.DataFrame()

        n_obj = self.n_objectives
        data = {f"f{i+1}": self.F[:, i] for i in range(n_obj)}

        if self.X is not None:
            n_var = self.X.shape[1]
            for i in range(n_var):
                data[f"x{i+1}"] = self.X[:, i]

        return pd.DataFrame(data)

    def save(self, path: str) -> None:
        """
        Save results to a directory (CSV files for F, X, and metadata).

        Args:
            path: Directory path to save results
        """
        import json
        from pathlib import Path

        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.F is not None:
            np.savetxt(out_dir / "FUN.csv", self.F, delimiter=",")
        if self.X is not None:
            np.savetxt(out_dir / "X.csv", self.X, delimiter=",")

        metadata = {
            "n_solutions": len(self),
            "n_objectives": self.n_objectives,
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Results saved to {out_dir}")

@dataclass
class OptimizeConfig:
    """
    Canonical configuration for a single optimization run.
    """

    problem: ProblemProtocol
    algorithm: str
    algorithm_config: Any
    termination: Tuple[str, Any]
    seed: int
    engine: str = "numpy"
    eval_backend: EvaluationBackend | str | None = None  # name or backend instance
    live_viz: Any = None


def _normalize_cfg(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(cfg)


def _available_algorithms() -> str:
    return ", ".join(sorted(ALGORITHMS.keys()))


def optimize(
    config: OptimizeConfig,
    *,
    engine: str | None = None,
) -> OptimizationResult:
    """
    Run a single optimization for the provided problem/config pair.

    Args:
        config: OptimizeConfig with problem, algorithm, and settings
        engine: Override backend engine ('numpy', 'numba', 'moocore').
                If provided, overrides config.engine.

    Returns:
        OptimizationResult with Pareto front and helper methods

    Examples:
        # Standard usage
        result = optimize(config)

        # Override engine at call time
        result = optimize(config, engine="numba")
    """
    if not isinstance(config, OptimizeConfig):
        raise TypeError("optimize() expects an OptimizeConfig instance.")
    cfg = config

    cfg_dict = _normalize_cfg(cfg.algorithm_config)
    algorithm_name = (cfg.algorithm or "").lower()
    if not algorithm_name:
        raise ValueError(
            "OptimizeConfig.algorithm must be specified. "
            f"Available algorithms: {_available_algorithms()}"
        )
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm_name}'. Available: {_available_algorithms()}")

    # Engine priority: function arg > config > algorithm_config > default
    effective_engine = engine or cfg.engine or cfg_dict.get("engine", "numpy")
    kernel = resolve_kernel(effective_engine) if effective_engine != "numpy" else NumPyKernel()

    if cfg.eval_backend is not None:
        backend = cfg.eval_backend
    else:
        backend_name = cfg_dict["eval_backend"] if isinstance(cfg_dict, dict) and "eval_backend" in cfg_dict else "serial"
        backend = resolve_eval_backend(backend_name)

    algo_ctor = resolve_algorithm(algorithm_name)
    algorithm = algo_ctor(cfg_dict, kernel=kernel)

    run_fn = getattr(algorithm, "run")
    sig = inspect.signature(run_fn)
    kwargs = {"problem": cfg.problem, "termination": cfg.termination, "seed": cfg.seed}
    if "eval_backend" in sig.parameters:
        kwargs["eval_backend"] = backend
    if "live_viz" in sig.parameters:
        kwargs["live_viz"] = cfg.live_viz
    result = run_fn(**kwargs)
    return OptimizationResult(result)


def run_optimization(
    problem: ProblemProtocol,
    algorithm: str = "nsgaii",
    *,
    max_evaluations: int = 10000,
    pop_size: int = 100,
    engine: str = "numpy",
    seed: int = 42,
    **kwargs: Any,
) -> OptimizationResult:
    """
    Simplified optimization interface - run without creating OptimizeConfig.

    This is a convenience function for quick experiments. For full control,
    use `optimize()` with `OptimizeConfig`.

    Args:
        problem: Problem instance to optimize
        algorithm: Algorithm name ('nsgaii', 'moead', 'spea2', 'smsemoa', 'nsgaiii')
        max_evaluations: Maximum function evaluations
        pop_size: Population size
        engine: Backend engine ('numpy', 'numba', 'moocore')
        seed: Random seed for reproducibility
        **kwargs: Additional algorithm-specific parameters

    Returns:
        OptimizationResult with Pareto front and helper methods

    Examples:
        from vamos import run_optimization, ZDT1

        problem = ZDT1(n_var=30)
        result = run_optimization(problem, "nsgaii", max_evaluations=5000)
        result.summary()
        result.plot()
    """
    from vamos.engine.algorithm.config import (
        NSGAIIConfig,
        MOEADConfig,
        SPEA2Config,
        SMSEMOAConfig,
        NSGAIIIConfig,
    )

    algorithm = algorithm.lower()
    n_var = getattr(problem, "n_var", None)

    # Build config based on algorithm
    if algorithm == "nsgaii":
        algo_config = NSGAIIConfig.default(pop_size=pop_size, n_var=n_var, engine=engine)
    elif algorithm == "moead":
        algo_config = MOEADConfig.default(pop_size=pop_size, n_var=n_var, engine=engine)
    elif algorithm == "spea2":
        algo_config = SPEA2Config.default(pop_size=pop_size, n_var=n_var, engine=engine)
    elif algorithm == "smsemoa":
        algo_config = SMSEMOAConfig.default(pop_size=pop_size, n_var=n_var, engine=engine)
    elif algorithm == "nsgaiii":
        n_obj = getattr(problem, "n_obj", 3)
        algo_config = NSGAIIIConfig.default(pop_size=pop_size, n_var=n_var, n_obj=n_obj, engine=engine)
    else:
        from vamos.exceptions import InvalidAlgorithmError
        raise InvalidAlgorithmError(
            algorithm,
            available=["nsgaii", "moead", "spea2", "smsemoa", "nsgaiii"],
        )

    config = OptimizeConfig(
        problem=problem,
        algorithm=algorithm,
        algorithm_config=algo_config,
        termination=("n_eval", max_evaluations),
        seed=seed,
        engine=engine,
    )

    return optimize(config)


__all__ = ["OptimizeConfig", "optimize", "OptimizationResult", "pareto_filter", "run_optimization"]
