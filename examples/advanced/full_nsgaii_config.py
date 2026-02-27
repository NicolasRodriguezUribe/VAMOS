

from typing import Any

import numpy as np

from vamos.algorithms import NSGAIIConfig
from vamos.experiment.unified import optimize
from pathlib import Path
from vamos.foundation.core.io_utils import write_population


def report_top_100_unbounded(entry: dict[str, Any]) -> None:
    result = entry["result"]
    try:
        top = result.top_k(k=100, source="archive", method="knee", nondominated_only=True)
    except ValueError:
        print("\n[unbounded_archive] No archive solutions available.")
        return

    top_f = np.asarray(top["F"], dtype=float)
    top_scores = np.asarray(top["scores"], dtype=float)
    top_x = top["X"]

    print(f"\n[unbounded_archive] Top {top_f.shape[0]} best compromise solutions")
    print("  rank | score | objectives")
    for i in range(top_f.shape[0]):
        print(f"  {i + 1:4d} | {top_scores[i]:.6f} | {np.array2string(top_f[i], precision=6, suppress_small=True)}")

    if top_x is not None and top_f.shape[0] > 0:
        print("\n  Best decision vector (rank=1):")
        print(f"  {np.array2string(np.asarray(top_x)[0], precision=6, suppress_small=True)}")

def save_results(result, out_dir: Path | None = None) -> Path:
    """Save Pareto front `FUN.csv` (and `X.csv` if available) to `out_dir`.

    If `out_dir` is None, the current working directory is used.
    Uses `write_population` with a NumPy fallback.
    Returns the `out_dir` path used.
    """
    out_dir = Path.cwd() if out_dir is None else Path(out_dir)
    try:
        write_population(out_dir, result.F, X=getattr(result, "X", None), G=getattr(result, "G", None))
        print(f"Saved FUN.csv (and X.csv if available) to {out_dir}")
        return out_dir
    except Exception:
        try:
            import numpy as _np

            out_dir.mkdir(parents=True, exist_ok=True)
            _np.savetxt(out_dir / "FUN.csv", result.F, delimiter=",")
            if getattr(result, "X", None) is not None:
                _np.savetxt(out_dir / "X.csv", result.X, delimiter=",")
            print(f"Saved FUN.csv (and X.csv if available) to {out_dir}")
            return out_dir
        except Exception as exc:
            print(f"Failed to save FUN.csv: {exc}")
            return out_dir



def main():
    config = (
        NSGAIIConfig.builder()
        .pop_size(100)
        .offspring_size(100)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", pressure=2)
        .external_archive(capacity=None)
        .build()
    )
    # print(config)

    result = optimize("dtlz2", algorithm="nsgaii", algorithm_config=config, max_evaluations=40000, seed=42, engine="numba")

    report_top_100_unbounded({"result": result})

    # Save Pareto front to FUN.csv in the current working directory
    save_results(result)

if __name__ == "__main__":
    main()