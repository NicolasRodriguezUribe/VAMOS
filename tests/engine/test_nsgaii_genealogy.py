from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


def test_nsgaii_track_genealogy_runs_and_returns_stats():
    pop_size = 8
    cfg = (
        NSGAIIConfig()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .engine("numpy")
        .track_genealogy(True)
        .fixed()
    )
    algorithm = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    result = algorithm.run(ZDT1Problem(n_var=6), termination=("n_eval", pop_size + 8), seed=0)

    genealogy = result.get("genealogy")
    assert genealogy is not None
    assert "operator_stats" in genealogy
    assert "generation_contributions" in genealogy
