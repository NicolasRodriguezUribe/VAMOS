from __future__ import annotations

from vamos.monitoring import HVConvergenceMonitor, HVConvergenceConfig


def test_hv_monitor_stops_on_plateau():
    cfg = HVConvergenceConfig(
        every_k=100,
        window=5,
        patience=3,
        epsilon=1e-6,
        epsilon_mode="abs",
        statistic="median",
        min_points=12,
        confidence=None,
        rng_seed=0,
    )
    mon = HVConvergenceMonitor(cfg)

    # Increase then plateau
    hv = [
        0.1,
        0.2,
        0.3,
        0.35,
        0.37,
        0.371,
        0.371,
        0.371,
        0.371,
        0.371,
        0.371,
        0.371,
        0.371,
        0.371,
        0.371,
    ]
    stop = False
    for i, v in enumerate(hv):
        dec = mon.add_sample(evals=100 * (i + 1), hv=v)
        if dec.stop:
            stop = True
            assert dec.reason in ("converged_hv", "already_stopped")
            break
    assert stop


def test_hv_monitor_does_not_stop_while_improving():
    cfg = HVConvergenceConfig(
        every_k=100,
        window=5,
        patience=4,
        epsilon=1e-3,
        epsilon_mode="rel",
        statistic="mean",
        min_points=12,
        confidence=None,
        rng_seed=0,
    )
    mon = HVConvergenceMonitor(cfg)
    hv = [0.1 + 0.01 * i for i in range(30)]
    for i, v in enumerate(hv):
        dec = mon.add_sample(evals=100 * (i + 1), hv=v)
        assert not dec.stop
