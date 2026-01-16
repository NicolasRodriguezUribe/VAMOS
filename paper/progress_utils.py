from __future__ import annotations

from contextlib import contextmanager
import sys
import time
from typing import Iterator


def _format_duration(seconds: float) -> str:
    if seconds != seconds or seconds == float("inf"):
        return "--:--:--"
    seconds_int = int(max(0.0, seconds))
    hours, rem = divmod(seconds_int, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ProgressBar:
    def __init__(
        self,
        *,
        total: int,
        desc: str,
        width: int = 28,
        min_interval_s: float = 0.2,
        file=None,
    ) -> None:
        self.total = max(0, int(total))
        self.desc = str(desc)
        self.width = max(8, int(width))
        self.min_interval_s = max(0.0, float(min_interval_s))
        self.file = sys.stderr if file is None else file

        self._count = 0
        self._start = time.perf_counter()
        self._last_render = 0.0
        self._last_len = 0

        if self.total > 0:
            self._render(force=True)

    @property
    def count(self) -> int:
        return self._count

    def update(self, n: int = 1) -> None:
        if self.total <= 0:
            return
        self._count = min(self.total, self._count + int(n))
        self._render()

    def _render(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and (now - self._last_render) < self.min_interval_s and self._count < self.total:
            return
        self._last_render = now

        elapsed = now - self._start
        rate = (self._count / elapsed) if elapsed > 0 else 0.0
        remaining = max(0, self.total - self._count)
        eta = (remaining / rate) if rate > 0 else float("inf")

        frac = (self._count / self.total) if self.total else 1.0
        filled = int(round(self.width * frac))
        filled = min(self.width, max(0, filled))
        bar = "#" * filled + "-" * (self.width - filled)

        pct = 100.0 * frac
        msg = f"{self.desc} [{bar}] {self._count}/{self.total} ({pct:5.1f}%) ETA {_format_duration(eta)}"
        pad = " " * max(0, self._last_len - len(msg))
        self.file.write("\r" + msg + pad)
        self.file.flush()
        self._last_len = len(msg)

    def close(self) -> None:
        if self.total <= 0:
            return
        self._render(force=True)
        self.file.write("\n")
        self.file.flush()


@contextmanager
def joblib_progress(*, total: int, desc: str) -> Iterator[ProgressBar]:
    """
    Display a simple progress bar for joblib.Parallel calls.

    Works by temporarily patching joblib.parallel.BatchCompletionCallBack so the
    bar updates as tasks complete (in the parent process).
    """
    bar = ProgressBar(total=total, desc=desc)
    try:
        import joblib.parallel
    except Exception:  # pragma: no cover
        try:
            yield bar
        finally:
            bar.close()
        return

    old_callback = joblib.parallel.BatchCompletionCallBack

    class _ProgressCallback(old_callback):  # type: ignore[misc]
        def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            bar.update(getattr(self, "batch_size", 1))
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = _ProgressCallback  # type: ignore[assignment]
    try:
        yield bar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback  # type: ignore[assignment]
        bar.close()

