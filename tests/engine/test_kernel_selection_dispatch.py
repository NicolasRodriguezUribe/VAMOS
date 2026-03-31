from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


class _TrackingNumPyKernel(NumPyKernel):
    def __init__(self) -> None:
        super().__init__()
        self.tournament_calls = 0

    def tournament_selection(self, ranks, crowding, pressure, rng, n_parents):  # noqa: ANN001 - kernel protocol signature
        self.tournament_calls += 1
        return super().tournament_selection(ranks, crowding, pressure, rng, n_parents)


def test_nsgaii_mating_dispatches_to_kernel_tournament_selection() -> None:
    pop_size = 8
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .offspring_size(pop_size)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    )
    kernel = _TrackingNumPyKernel()
    algorithm = NSGAII(cfg.to_dict(), kernel=kernel)
    problem = ZDT1Problem(n_var=6)

    algorithm.run(problem, termination=("max_evaluations", pop_size * 2), seed=11)

    assert kernel.tournament_calls > 0
