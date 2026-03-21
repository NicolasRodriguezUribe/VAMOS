from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocSmokeCase:
    name: str
    source_path: str
    code: str


DOC_SMOKE_CASES = [
    DocSmokeCase(
        name="readme_quickstart",
        source_path="README.md",
        code="""
from vamos import optimize

result = optimize(
    "zdt1",
    algorithm="nsgaii",
    max_evaluations=200,
    pop_size=40,
    engine="numpy",
    seed=42,
)
front = result.front()
assert front is not None
assert len(front) > 0
""",
    ),
    DocSmokeCase(
        name="readme_make_problem",
        source_path="README.md",
        code="""
from vamos import make_problem, optimize

problem = make_problem(
    lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
    n_var=2,
    n_obj=2,
    bounds=[(0, 1), (0, 1)],
    encoding="real",
)
result = optimize(problem, algorithm="nsgaii", max_evaluations=200, seed=42)
assert result.F is not None
assert result.F.shape[1] == 2
""",
    ),
    DocSmokeCase(
        name="getting_started_config_object",
        source_path="docs/guide/getting-started.md",
        code="""
from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.problems import ZDT1

problem = ZDT1(n_var=10)
algo_cfg = NSGAIIConfig.default(pop_size=40, n_var=problem.n_var)
result = optimize(
    problem,
    algorithm="nsgaii",
    algorithm_config=algo_cfg,
    max_evaluations=200,
    seed=42,
    engine="numpy",
)
assert result.F is not None
assert result.F.shape[1] == problem.n_obj
""",
    ),
]
