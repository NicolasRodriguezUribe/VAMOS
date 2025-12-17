import pytest
import numpy as np

pytest.importorskip("matplotlib")

from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.ux.visualization.live_viz import LiveVisualization, LiveParetoPlot


class DummyProblem:
    def __init__(self):
        self.n_var = 2
        self.n_obj = 2
        self.xl = 0.0
        self.xu = 1.0
        self.encoding = "real"

    def evaluate(self, X, out):
        f1 = X[:, 0]
        f2 = 1.0 - X[:, 1]
        out["F"] = np.stack([f1, f2], axis=1)


class RecorderViz(LiveVisualization):
    def __init__(self):
        self.starts = 0
        self.gens = 0
        self.ends = 0

    def on_start(self, problem=None, algorithm=None, config=None):
        self.starts += 1

    def on_generation(self, generation: int, F=None, X=None, stats=None):
        self.gens += 1

    def on_end(self, final_F=None, final_stats=None):
        self.ends += 1


def test_live_viz_callbacks_invoked(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    cfg = (
        NSGAIIConfig()
        .pop_size(6)
        .offspring_size(6)
        .crossover("sbx", prob=0.9, eta=10.0)
        .mutation("pm", prob="1/n", eta=10.0)
        .selection("tournament", pressure=2)
        .survival("nsga2")
        .engine("numpy")
        .fixed()
    )
    algo = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    problem = DummyProblem()
    recorder = RecorderViz()

    result = algo.run(problem, termination=("n_eval", 12), seed=1, live_viz=recorder)

    assert result["F"].shape[0] == cfg.pop_size
    assert recorder.starts == 1
    assert recorder.ends == 1
    assert recorder.gens >= 1


def test_live_pareto_plot_saves_file(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    viz = LiveParetoPlot(save_final_path=tmp_path / "live.png", update_interval=1, max_points=10)
    viz.on_start()
    F = np.array([[0.1, 0.9], [0.2, 0.8], [0.5, 0.5]])
    viz.on_generation(0, F=F)
    viz.on_end(final_F=F)
    assert (tmp_path / "live.png").exists()
