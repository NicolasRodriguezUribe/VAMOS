import pytest

from vamos import OptimizationResult, optimize


@pytest.mark.parametrize(
    ("algorithm", "problem", "pop_size", "budget"),
    [
        ("nsgaii", "zdt1", 12, 36),
        ("nsgaii", "zdt2", 12, 36),
        ("smsemoa", "zdt1", 10, 24),
        ("moead", "zdt2", 12, 36),
    ],
)
def test_optimize_algorithm_problem_matrix(
    algorithm: str,
    problem: str,
    pop_size: int,
    budget: int,
) -> None:
    result = optimize(
        problem,
        algorithm=algorithm,
        max_evaluations=budget,
        pop_size=pop_size,
        seed=7,
        engine="numpy",
    )

    assert isinstance(result, OptimizationResult)
    assert result.F is not None
    assert result.F.shape[1] == 2
    assert result.F.shape[0] > 0
