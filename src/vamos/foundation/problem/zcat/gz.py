from __future__ import annotations

import math
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

_DBL_EPSILON = 2.220446049250313e-16
_PI = math.pi
_E = math.e
FloatArray: TypeAlias = npt.NDArray[np.float64]


def _empty(size: int) -> FloatArray:
    return cast(FloatArray, np.empty(size, dtype=float))


def zcat_fix_to_01(a: float) -> float:
    """Clamp values very close to [0, 1] endpoints to avoid roundoff drift."""
    if a <= 0.0 and a >= -_DBL_EPSILON:
        return 0.0
    if a >= 1.0 and a <= 1.0 + _DBL_EPSILON:
        return 1.0
    return a


def zcat_lq(y: float, z: float) -> bool:
    """Numerically robust ``y <= z`` check used in the original reference."""
    return y < z or abs(z - y) < _DBL_EPSILON


def zcat_value_in(y: float, lb: float, ub: float) -> bool:
    """Check interval membership with inclusive bounds and tolerance."""
    return zcat_lq(lb, y) and zcat_lq(y, ub)


def zcat_for_all_value_in(y: FloatArray, m: int, lb: float, ub: float) -> bool:
    """Check ``lb <= y_i <= ub`` for all ``i in [0, m)``."""
    limit = min(max(m, 0), int(y.shape[0]))
    for i in range(limit):
        if not zcat_value_in(float(y[i]), lb, ub):
            return False
    return True


def theta_j(j: int, m: int, n: int) -> float:
    """Angular offset used in ZCAT g-functions."""
    return 2.0 * _PI * (j - 1.0) / (n - m)


def g0(y: FloatArray, m: int, n: int) -> FloatArray:
    size = n - m
    g = _empty(size)
    g.fill(0.2210)
    return g


def g1(y: FloatArray, m: int, n: int) -> FloatArray:
    size = n - m
    g = _empty(size)
    for j in range(1, size + 1):
        total = 0.0
        angle = theta_j(j, m, n)
        for i in range(1, m + 1):
            total += math.sin(1.5 * _PI * float(y[i - 1]) + angle)
        g[j - 1] = total / (2.0 * m) + 0.5
    return g


def g2(y: FloatArray, m: int, n: int) -> FloatArray:
    size = n - m
    g = _empty(size)
    for j in range(1, size + 1):
        total = 0.0
        angle = theta_j(j, m, n)
        for i in range(1, m + 1):
            yi = float(y[i - 1])
            total += (yi**2.0) * math.sin(4.5 * _PI * yi + angle)
        g[j - 1] = total / (2.0 * m) + 0.5
    return g


def g3(y: FloatArray, m: int, n: int) -> FloatArray:
    size = n - m
    g = _empty(size)
    for j in range(1, size + 1):
        total = 0.0
        angle = theta_j(j, m, n)
        for i in range(1, m + 1):
            total += math.cos(_PI * float(y[i - 1]) + angle) ** 2.0
        g[j - 1] = total / m
    return g


def g4(y: FloatArray, m: int, n: int) -> FloatArray:
    size = n - m
    g = _empty(size)
    mu = float(np.sum(y[:m])) / m
    for j in range(1, size + 1):
        g[j - 1] = (mu / 2.0) * math.cos(4.0 * _PI * mu + theta_j(j, m, n)) + 0.5
    return g


def g5(y: FloatArray, m: int, n: int) -> FloatArray:
    size = n - m
    g = _empty(size)
    for j in range(1, size + 1):
        total = 0.0
        angle = theta_j(j, m, n)
        for i in range(1, m + 1):
            total += math.sin(2.0 * _PI * float(y[i - 1]) - 1.0 + angle) ** 3.0
        g[j - 1] = total / (2.0 * m) + 0.5
    return g


def g6(y: FloatArray, m: int, n: int) -> FloatArray:
    size = n - m
    g = _empty(size)
    denom = -10.0 * math.exp(-2.0 / 5.0) - math.exp(-1.0) + 10.0 + _E
    for j in range(1, size + 1):
        s1 = 0.0
        s2 = 0.0
        angle = theta_j(j, m, n)
        for i in range(1, m + 1):
            yi = float(y[i - 1])
            s1 += yi**2.0
            s2 += math.cos(11.0 * _PI * yi + angle) ** 3.0
        s1 /= m
        s2 /= m
        numer = -10.0 * math.exp((-2.0 / 5.0) * math.sqrt(s1)) - math.exp(s2) + 10.0 + _E
        g[j - 1] = numer / denom
    return g


def g7(y: FloatArray, m: int, n: int) -> FloatArray:
    size = n - m
    g = _empty(size)
    mu = float(np.sum(y[:m])) / m
    denom = 1.0 + _E - math.exp(-1.0)
    for j in range(1, size + 1):
        angle = theta_j(j, m, n)
        g[j - 1] = (mu + math.exp(math.sin(7.0 * _PI * mu - _PI / 2.0 + angle)) - math.exp(-1.0)) / denom
    return g


def g8(y: FloatArray, m: int, n: int) -> FloatArray:
    size = n - m
    g = _empty(size)
    for j in range(1, size + 1):
        total = 0.0
        angle = theta_j(j, m, n)
        for i in range(1, m + 1):
            total += abs(math.sin(2.5 * _PI * (float(y[i - 1]) - 0.5) + angle))
        g[j - 1] = total / m
    return g


def g9(y: FloatArray, m: int, n: int) -> FloatArray:
    size = n - m
    g = _empty(size)
    mu = float(np.sum(y[:m])) / m
    for j in range(1, size + 1):
        total = 0.0
        angle = theta_j(j, m, n)
        for i in range(1, m + 1):
            total += abs(math.sin(2.5 * _PI * float(y[i - 1]) - _PI / 2.0 + angle))
        g[j - 1] = mu / 2.0 - total / (2.0 * m) + 0.5
    return g


def g10(y: FloatArray, m: int, n: int) -> FloatArray:
    size = n - m
    g = _empty(size)
    denom = 2.0 * (m**3.0)
    for j in range(1, size + 1):
        total = 0.0
        angle = theta_j(j, m, n)
        for i in range(1, m + 1):
            total += math.sin((4.0 * float(y[i - 1]) - 2.0) * _PI + angle)
        g[j - 1] = (total**3.0) / denom + 0.5
    return g


def evaluate_g(g_function_id: int, y: FloatArray, m: int, n: int) -> FloatArray:
    if g_function_id == 0:
        return g0(y, m, n)
    if g_function_id == 1:
        return g1(y, m, n)
    if g_function_id == 2:
        return g2(y, m, n)
    if g_function_id == 3:
        return g3(y, m, n)
    if g_function_id == 4:
        return g4(y, m, n)
    if g_function_id == 5:
        return g5(y, m, n)
    if g_function_id == 6:
        return g6(y, m, n)
    if g_function_id == 7:
        return g7(y, m, n)
    if g_function_id == 8:
        return g8(y, m, n)
    if g_function_id == 9:
        return g9(y, m, n)
    if g_function_id == 10:
        return g10(y, m, n)
    raise ValueError(f"Unsupported g-function id: {g_function_id}")


def z1(j_values: FloatArray) -> float:
    j_size = int(j_values.shape[0])
    return (10.0 / j_size) * float(np.sum(j_values * j_values))


def z2(j_values: FloatArray) -> float:
    return 10.0 * float(np.max(np.abs(j_values)))


def z3(j_values: FloatArray) -> float:
    j_size = int(j_values.shape[0])
    k = 5.0
    total = 0.0
    for value in j_values:
        total += (float(value) ** 2.0 - math.cos((2.0 * k - 1.0) * _PI * float(value)) + 1.0) / 3.0
    return (10.0 / j_size) * total


def z4(j_values: FloatArray) -> float:
    j_size = int(j_values.shape[0])
    k = 5.0
    pow1 = float(np.max(np.abs(j_values)))
    pow2 = 0.0
    for value in j_values:
        pow2 += 0.5 * (math.cos((2.0 * k - 1.0) * _PI * float(value)) + 1.0)
    numer = math.exp(pow1**0.5) - math.exp(pow2 / j_size) - 1.0 + _E
    return (10.0 / (2.0 * _E - 2.0)) * numer


def z5(j_values: FloatArray) -> float:
    j_size = int(j_values.shape[0])
    total = 0.0
    for value in j_values:
        total += abs(float(value)) ** 0.002
    return -0.7 * z3(j_values) + (10.0 / j_size) * total


def z6(j_values: FloatArray) -> float:
    j_size = int(j_values.shape[0])
    total = 0.0
    for value in j_values:
        total += abs(float(value))
    return float(-0.7 * z4(j_values) + 10.0 * (total / j_size) ** 0.002)


def zbias(z_value: float) -> float:
    return float(abs(z_value) ** 0.05)


def zcat_get_j(objective_index: int, number_of_objectives: int, w: FloatArray, w_size: int) -> FloatArray:
    values: list[float] = []
    for j in range(1, w_size + 1):
        if (j - objective_index) % number_of_objectives == 0:
            values.append(float(w[j - 1]))
    if not values:
        values.append(float(w[0]))
    result = _empty(len(values))
    for idx, value in enumerate(values):
        result[idx] = value
    return result


def zcat_evaluate_z(j_values: FloatArray, objective_index: int, imbalance: bool, level: int) -> float:
    if imbalance:
        return z4(j_values) if objective_index % 2 == 0 else z1(j_values)

    if level == 1:
        return z1(j_values)
    if level == 2:
        return z2(j_values)
    if level == 3:
        return z3(j_values)
    if level == 4:
        return z4(j_values)
    if level == 5:
        return z5(j_values)
    if level == 6:
        return z6(j_values)
    return z1(j_values)


__all__ = [
    "evaluate_g",
    "zcat_evaluate_z",
    "zcat_fix_to_01",
    "zcat_for_all_value_in",
    "zcat_get_j",
    "zcat_value_in",
    "zbias",
]
