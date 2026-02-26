"""
Run loop and checkpoint helpers for NSGA-II.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from vamos.engine.algorithm.components.hooks import live_should_stop
from vamos.engine.hooks.live_viz import LiveVisualization, NoOpLiveVisualization
from vamos.foundation.eval.backends import EvaluationBackend

from .ask_tell import _build_fused_operator_params, _is_real_sbx_pm_pipeline
from .setup import initialize_run
from .state import build_result, finalize_genealogy, get_archive_contents

if TYPE_CHECKING:
    from .nsgaii import NSGAII


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


#: Default number of generations between checkpoint saves.
DEFAULT_CHECKPOINT_INTERVAL: int = 50


def notify_generation(
    algo: NSGAII,
    live_cb: LiveVisualization,
    generation: int,
    F: np.ndarray,
    problem: Any | None = None,
    evals: int | None = None,
) -> bool:
    """Notify live visualization of generation progress."""
    st = algo._st
    nd_mask = None
    ranks = None
    if st is not None and st.immigration_manager is not None and problem is not None:
        changed = bool(
            st.immigration_manager.apply_generation(
                generation=int(generation),
                state=st,
                problem=problem,
                kernel=algo.kernel,
            )
        )
        if changed:
            F = st.F
            if st.steady_state:
                st.ranks = None
                st.crowding = None
                st.fronts = None

    try:
        ranks, _ = algo.kernel.nsga2_ranking(F)
        nd_mask = ranks == ranks.min(initial=0)
    except (ValueError, IndexError) as exc:
        _logger().debug("Failed to compute non-dominated front for viz: %s", exc)
        nd_mask = None

    stats: dict[str, Any] = {"evals": int(evals)} if evals is not None else {}
    if st is not None and st.immigration_manager is not None:
        ig = st.immigration_manager.stats_for_generation(int(generation))
        stats["immigration"] = {
            "events": int(ig.events),
            "inserted": int(ig.inserted),
            "replaced_indices": [int(i) for i in (ig.replaced_indices or [])],
            "replaced_pages": [int(p) for p in (ig.replaced_pages or [])],
            "mating_participation": int(ig.mating_participation),
        }

    live_mode = str(getattr(st, "live_callback_mode", "nd_only")).lower() if st is not None else "nd_only"
    if nd_mask is None:
        live_F = F
        live_X = st.X if st is not None else None
    elif live_mode in {"population", "population_archive"}:
        live_F = F
        live_X = st.X if st is not None else None
    else:
        live_F = F[nd_mask]
        live_X = st.X[nd_mask] if st is not None else None

    if st is not None and live_mode == "population_archive":
        archive_payload = get_archive_contents(st)
        if archive_payload is not None:
            stats["archive"] = {
                "X": np.asarray(archive_payload.get("X")),
                "F": np.asarray(archive_payload.get("F")),
            }

    if stats:
        live_cb.on_generation(generation, F=live_F, X=live_X, stats=stats)
    else:
        live_cb.on_generation(generation, F=live_F, X=live_X, stats=None)

    if st is not None and callable(st.generation_callback):
        try:
            copy_arrays = bool(st.generation_callback_copy)
            payload: dict[str, Any] = {
                "generation": int(generation),
                "evaluations": int(evals) if evals is not None else None,
                "population": {
                    "X": np.asarray(st.X).copy() if copy_arrays else np.asarray(st.X),
                    "F": np.asarray(st.F).copy() if copy_arrays else np.asarray(st.F),
                    "G": (
                        np.asarray(st.G).copy()
                        if copy_arrays and isinstance(st.G, np.ndarray)
                        else np.asarray(st.G)
                        if isinstance(st.G, np.ndarray)
                        else None
                    ),
                },
                "nondominated": None,
                "archive": None,
                "stats": stats or None,
            }
            if nd_mask is not None:
                payload["nondominated"] = {
                    "X": (np.asarray(st.X[nd_mask]).copy() if copy_arrays else np.asarray(st.X[nd_mask])),
                    "F": (np.asarray(st.F[nd_mask]).copy() if copy_arrays else np.asarray(st.F[nd_mask])),
                }
            archive_payload = get_archive_contents(st)
            if archive_payload is not None:
                x_arch = np.asarray(archive_payload.get("X"))
                f_arch = np.asarray(archive_payload.get("F"))
                payload["archive"] = {
                    "X": x_arch.copy() if copy_arrays else x_arch,
                    "F": f_arch.copy() if copy_arrays else f_arch,
                }
            callback_stop = bool(st.generation_callback(payload))
            if callback_stop:
                return True
        except Exception as exc:
            _logger().debug("generation_callback failed: %s", exc)

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
        extra={
            "step": st.step,
            "replacements": st.replacements,
        },
    )
    _logger().info("Checkpoint saved: %s", path)


def _can_use_nsga2_evolve_fastpath(
    algo: NSGAII,
    st: Any,
    live_cb: LiveVisualization,
    hv_tracker: Any,
    checkpoint_dir: str | None,
) -> bool:
    if str(os.environ.get("VAMOS_ENABLE_CPP_EVOLVE_FASTPATH", "0")).strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    nsga2_evolve = getattr(algo.kernel, "nsga2_evolve", None)
    if not callable(nsga2_evolve):
        return False
    if st.kernel_profiler is not None:
        return False
    if not isinstance(live_cb, NoOpLiveVisualization):
        return False
    if st.steady_state:
        return False
    if st.G is not None:
        return False
    if st.offspring_size != st.pop_size:
        return False
    if st.track_genealogy:
        return False
    if st.aos_controller is not None:
        return False
    if st.immigration_manager is not None:
        return False
    if callable(st.parent_selection_filter):
        return False
    if st.non_breeding_indices.size > 0:
        return False
    if st.archive_manager is not None or st.result_archive is not None:
        return False
    if checkpoint_dir is not None:
        return False
    if st.generation_callback is not None:
        return False
    if bool(getattr(hv_tracker, "enabled", False)):
        return False
    if not _is_real_sbx_pm_pipeline(st):
        return False
    return True


def run_nsgaii(
    algo: NSGAII,
    problem: Any,
    termination: tuple[str, Any],
    seed: int,
    eval_strategy: EvaluationBackend | None = None,
    live_viz: LiveVisualization | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the NSGA-II algorithm."""
    import signal

    live_cb, eval_strategy, max_eval, n_eval, hv_tracker = initialize_run(
        algo,
        problem,
        termination,
        seed,
        eval_strategy,
        live_viz,
        checkpoint=checkpoint,
    )
    st = algo._st
    assert st is not None, "State not initialized"

    interrupted = False
    main_thread_signals = threading.current_thread() is threading.main_thread()
    original_handler = signal.getsignal(signal.SIGINT) if main_thread_signals else None

    def _handle_interrupt(signum: int, frame: Any | None) -> None:
        nonlocal interrupted
        interrupted = True
        _logger().info("Interrupt received, finishing current generation...")

    if main_thread_signals:
        signal.signal(signal.SIGINT, _handle_interrupt)

    generation = st.generation
    step = st.step
    replacements = st.replacements
    stop_requested = notify_generation(
        algo,
        live_cb,
        generation,
        st.F,
        problem=problem,
        evals=n_eval,
    )
    hv_reached = hv_tracker.enabled and hv_tracker.reached(st.hv_points_fn())
    prof = st.kernel_profiler

    def _measure(label: str) -> Any:
        return prof.measure(label) if prof is not None else nullcontext()

    try:
        used_fastpath = False
        if _can_use_nsga2_evolve_fastpath(algo, st, live_cb, hv_tracker, checkpoint_dir) and not stop_requested and not interrupted:
            remaining = max_eval - n_eval
            n_gen = int(remaining // st.offspring_size)
            if n_gen > 0:
                params = _build_fused_operator_params(st)

                def _eval_fn(X_off: np.ndarray) -> Any:
                    return eval_strategy.evaluate(X_off, problem)

                x_new, f_new = algo.kernel.nsga2_evolve(
                    st.X,
                    st.F,
                    _eval_fn,
                    n_gen,
                    params,
                    st.rng,
                    st.variation.xl,
                    st.variation.xu,
                )
                st.X = np.asarray(x_new)
                st.F = np.asarray(f_new)
                st.G = None
                added = int(n_gen * st.offspring_size)
                n_eval += added
                replacements += added
                step += n_gen
                generation += n_gen
                st.n_eval = n_eval
                st.replacements = replacements
                st.step = step
                st.generation = generation
                used_fastpath = True

        while not used_fastpath and n_eval < max_eval and not hv_reached and not stop_requested and not interrupted:
            loop_start_ns = time.perf_counter_ns() if prof is not None else 0
            if prof is not None:
                prof.start_generation(generation=generation, evaluations=n_eval)
            try:
                st.generation = generation
                st.step = step
                st.replacements = replacements
                X_off = algo.ask()
                with _measure("evaluation"):
                    eval_off = eval_strategy.evaluate(X_off, problem)
                hv_reached = algo.tell(eval_off)
                n_eval += X_off.shape[0]
                replacements += X_off.shape[0]
                st.n_eval = n_eval

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
                        stop_requested = stop_requested or notify_generation(
                            algo,
                            live_cb,
                            generation,
                            st.F,
                            problem=problem,
                            evals=n_eval,
                        )
                        if hv_tracker.enabled and hv_tracker.reached(st.hv_points_fn()):
                            hv_reached = True

                        if checkpoint_dir and generation % checkpoint_interval == 0:
                            save_checkpoint(algo, checkpoint_dir, seed, generation, n_eval)
                else:
                    generation += 1
                    st.generation = generation
                    stop_requested = notify_generation(
                        algo,
                        live_cb,
                        generation,
                        st.F,
                        problem=problem,
                        evals=n_eval,
                    )
                    if hv_tracker.enabled and hv_tracker.reached(st.hv_points_fn()):
                        hv_reached = True

                    if checkpoint_dir and generation % checkpoint_interval == 0:
                        save_checkpoint(algo, checkpoint_dir, seed, generation, n_eval)
            finally:
                if prof is not None:
                    prof.add_ns("generation_total", time.perf_counter_ns() - loop_start_ns)
                    prof.end_generation(generation=st.generation, evaluations=n_eval)
    finally:
        if main_thread_signals and original_handler is not None:
            signal.signal(signal.SIGINT, original_handler)

    result = build_result(st, n_eval, hv_reached, kernel=algo.kernel)
    result["interrupted"] = interrupted
    result["checkpoint"] = {
        "version": 1,
        "algorithm": "nsgaii",
        "X": st.X,
        "F": st.F,
        "G": st.G,
        "generation": st.generation,
        "n_eval": n_eval,
        "rng_state": cast(dict[str, Any], st.rng.bit_generator.state),
        "archive_X": st.archive_X,
        "archive_F": st.archive_F,
        "extra": {
            "step": st.step,
            "replacements": st.replacements,
        },
    }
    live_cb.on_end(final_F=st.F)
    finalize_genealogy(result, st, algo.kernel)
    return result


__all__ = ["run_nsgaii", "save_checkpoint", "notify_generation"]
