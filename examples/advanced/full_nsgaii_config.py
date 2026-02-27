

from vamos.algorithms import NSGAIIConfig
from vamos.experiment.unified import optimize
from pathlib import Path
import os
from vamos.foundation.core.io_utils import write_population


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
        .crossover(
            "blx_alpha",
            prob=0.9,
            alpha=0.5,
        )
        .repair("random")
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", k=2)
        .build()
    )
    # print(config)

    result = optimize("zdt4", algorithm="nsgaii", algorithm_config=config, max_evaluations=20000, seed=42)

    # Save Pareto front to FUN.csv in the current working directory
    save_results(result)

if __name__ == "__main__":
    main()