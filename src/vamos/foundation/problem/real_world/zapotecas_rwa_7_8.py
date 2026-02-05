from __future__ import annotations

import numpy as np


def _rwa7_xcc(alpha: np.ndarray, dha: np.ndarray, doa: np.ndarray, optt: np.ndarray) -> np.ndarray:
    val = (
        0.153
        - 0.322 * alpha
        + 0.396 * dha
        + 0.424 * doa
        + 0.0226 * optt
        + 0.175 * alpha * alpha
        + 0.0185 * dha * alpha
        - 0.0701 * dha * dha
        - 0.251 * doa * alpha
        + 0.179 * doa * dha
        + 0.0150 * doa * doa
        + 0.0134 * optt * alpha
        + 0.0296 * optt * dha
        + 0.0752 * optt * doa
        + 0.0192 * optt * optt
    )
    return np.asarray(val, dtype=float)


def _rwa7_tfmax(alpha: np.ndarray, dha: np.ndarray, doa: np.ndarray, optt: np.ndarray) -> np.ndarray:
    val = (
        0.692
        + 0.477 * alpha
        - 0.687 * dha
        - 0.080 * doa
        - 0.0650 * optt
        - 0.167 * alpha * alpha
        - 0.0129 * dha * alpha
        + 0.0796 * dha * dha
        - 0.0634 * doa * alpha
        - 0.0257 * doa * dha
        + 0.0877 * doa * doa
        - 0.0521 * optt * alpha
        + 0.00156 * optt * dha
        + 0.00198 * optt * doa
        + 0.0184 * optt * optt
    )
    return np.asarray(val, dtype=float)


def _rwa7_ttmax(alpha: np.ndarray, dha: np.ndarray, doa: np.ndarray, optt: np.ndarray) -> np.ndarray:
    val = (
        0.370
        - 0.205 * alpha
        + 0.0307 * dha
        + 0.108 * doa
        + 1.019 * optt
        - 0.135 * alpha * alpha
        + 0.0141 * dha * alpha
        + 0.0998 * dha * dha
        + 0.208 * doa * alpha
        - 0.0301 * doa * dha
        - 0.226 * doa * doa
        + 0.353 * optt * alpha
        - 0.0497 * optt * doa
        - 0.423 * optt * optt
        + 0.202 * dha * alpha * alpha
        - 0.281 * doa * alpha * alpha
        - 0.342 * dha * dha * alpha
        - 0.245 * dha * dha * doa
        + 0.281 * doa * doa * dha
        - 0.184 * optt * optt * alpha
        + 0.281 * dha * alpha * doa
    )
    return np.asarray(val, dtype=float)


class RWA7Problem:
    """
    Zapotecas-Martínez et al. (2023) RWA7: Liquid-rocket single element injector design.

    Objectives (minimize): Xcc, TFmax, TTmax (paper, Eq. 8).
    """

    def __init__(self) -> None:
        self.n_var = 4
        self.n_obj = 3
        self.encoding = "continuous"
        # alpha, dHA, dOA, OPTT; with X'' = 0.01 in => OPTT in [0.01, 0.02]
        self.xl = np.array([0.0, 0.0, -40.0, 0.01], dtype=float)
        self.xu = np.array([20.0, 25.0, 0.0, 0.02], dtype=float)

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        alpha, dha, doa, optt = (X[:, i] for i in range(4))

        xcc = _rwa7_xcc(alpha, dha, doa, optt)
        tfmax = _rwa7_tfmax(alpha, dha, doa, optt)
        ttmax = _rwa7_ttmax(alpha, dha, doa, optt)

        F = out["F"]
        F[:, 0] = xcc
        F[:, 1] = tfmax
        F[:, 2] = ttmax


class RWA8Problem:
    """
    Zapotecas-Martínez et al. (2023) RWA8: Liquid-rocket single element injector design.

    Objectives (minimize): TFmax, TW4, TTmax, Xcc (paper, Eq. 9).
    """

    def __init__(self) -> None:
        self.n_var = 4
        self.n_obj = 4
        self.encoding = "continuous"
        self.xl = np.array([0.0, 0.0, -40.0, 0.01], dtype=float)
        self.xu = np.array([20.0, 25.0, 0.0, 0.02], dtype=float)

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        alpha, dha, doa, optt = (X[:, i] for i in range(4))

        tfmax = _rwa7_tfmax(alpha, dha, doa, optt)
        tw4 = (
            0.758
            + 0.358 * alpha
            - 0.807 * dha
            + 0.0925 * doa
            - 0.0468 * optt
            - 0.172 * alpha * alpha
            + 0.0106 * dha * alpha
            + 0.0697 * dha * dha
            - 0.146 * doa * alpha
            - 0.0416 * doa * dha
            + 0.102 * doa * doa
            - 0.0694 * optt * alpha
            - 0.00503 * optt * dha
            + 0.0151 * optt * doa
            + 0.0173 * optt * optt
        )
        ttmax = _rwa7_ttmax(alpha, dha, doa, optt)
        xcc = _rwa7_xcc(alpha, dha, doa, optt)

        F = out["F"]
        F[:, 0] = tfmax
        F[:, 1] = tw4
        F[:, 2] = ttmax
        F[:, 3] = xcc


__all__ = ["RWA7Problem", "RWA8Problem"]
