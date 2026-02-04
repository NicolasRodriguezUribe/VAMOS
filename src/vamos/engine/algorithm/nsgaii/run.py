"""
Run loop and checkpoint helpers for NSGA-II.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING, cast

import numpy as np

from vamos.engine.algorithm.components.hooks import live_should_stop
from vamos.foundation.eval.backends import EvaluationBackend
from vamos.hooks.live_viz import LiveVisualization

from .setup import initialize_run
from .state import build_result, finalize_genealogy

if TYPE_CHECKING:
    from .nsgaii import NSGAII


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def notify_generation(
    algo: NSGAII,
    live_cb: LiveVisualization,
    generation: int,
    F: np.ndarray,
    evals: int | None = None,
) -> bool:
    """Notify live visualization of generation progress."""
    try:
        ranks, _ = algo.kernel.nsga2_ranking(F)
        nd_mask = ranks == ranks.min(initial=0)
        stats = {"evals": int(evals)} if evals is not None else None
        live_cb.on_generation(generation, F=F[nd_mask], stats=stats)
    except (ValueError, IndexError) as exc:
        _logger().debug("Failed to compute non-dominated front for viz: %s", exc)
        stats = {"evals": int(evals)} if evals is not None else None
        live_cb.on_generation(generation, F=F, stats=stats)
    return live_should_stop(live_cb)


def save_checkpoint(algo: NSGAII, checkpoint_dir: str, seed: int, generation: int, n_eval: int) -> None:
    """Save current state to checkpoint file."""
    from pathlib import Path

    from vamos.foundation.checkpoint import save_checkpoint

    st = algo._st
    if st is None:
        return

    path = Path(checkpoint_dir) / f"nsgaii_seed{seed}_gen{generation}.ckpt"
    save_checkpoint(
        path,
        X=st.X,
        F=st.F,
        generation=generation,
        n_eval=n_eval,
        rng_state=cast(dict[str, Any], st.rng.bit_generator.state),
        G=st.G,
        archive_X=st.archive_X,
        archive_F=st.archive_F,
    )
    _logger().info("Checkpoint saved: %s", path)


def run_nsgaii(
    algo: NSGAII,
    problem: Any,
    termination: tuple[str, Any],
    seed: int,
    eval_strategy: EvaluationBackend | None = None,
    live_viz: LiveVisualization | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = 50,
) -> dict[str, Any]:
    """Run the NSGA-II algorithm."""
    import signal

    live_cb, eval_strategy, max_eval, n_eval, hv_tracker = initialize_run(algo, problem, termination, seed, eval_strategy, live_viz)
    st = algo._st
    assert st is not None, "State not initialized"

    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def _handle_interrupt(signum: int, frame: Any | None) -> None:
        nonlocal interrupted
        interrupted = True
        _logger().info("Interrupt received, finishing current generation...")

    signal.signal(signal.SIGINT, _handle_interrupt)

    generation = 0
    step = 0
    replacements = 0
    stop_requested = notify_generation(algo, live_cb, generation, st.F, evals=n_eval)
    hv_reached = hv_tracker.enabled and hv_tracker.reached(st.hv_points_fn())

    try:
        while n_eval < max_eval and not hv_reached and not stop_requested and not interrupted:
            st.generation = generation
            st.step = step
            st.replacements = replacements
            X_off = algo.ask()
            eval_off = eval_strategy.evaluate(X_off, problem)
            hv_reached = algo.tell(eval_off, st.pop_size)
            n_eval += X_off.shape[0]
            replacements += X_off.shape[0]

            step += 1
            st.step = step
            st.replacements = replacements

            if st.steady_state:
                if not stop_requested:
                    stop_requested = live_should_stop(live_cb)
                new_generation = replacements // st.pop_size
                if new_generation != generation:
                    generation = new_generation
                    st.generation = generation
                    stop_requested = stop_requested or notify_generation(algo, live_cb, generation, st.F, evals=n_eval)
                    if hv_tracker.enabled and hv_tracker.reached(st.hv_points_fn()):
                        hv_reached = True

                    if checkpoint_dir and generation % checkpoint_interval == 0:
                        save_checkpoint(algo, checkpoint_dir, seed, generation, n_eval)
            else:
                generation += 1
                st.generation = generation
                stop_requested = notify_generation(algo, live_cb, generation, st.F, evals=n_eval)
                if hv_tracker.enabled and hv_tracker.reached(st.hv_points_fn()):
                    hv_reached = True

                if checkpoint_dir and generation % checkpoint_interval == 0:
                    save_checkpoint(algo, checkpoint_dir, seed, generation, n_eval)
    finally:
        signal.signal(signal.SIGINT, original_handler)

    result = build_result(st, n_eval, hv_reached, kernel=algo.kernel)
    result["interrupted"] = interrupted
    live_cb.on_end(final_F=st.F)
    finalize_genealogy(result, st, algo.kernel)
    return result


__all__ = ["run_nsgaii", "save_checkpoint", "notify_generation"]
