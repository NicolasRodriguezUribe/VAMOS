from __future__ import annotations

import math
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from .gz import zcat_fix_to_01, zcat_for_all_value_in, zcat_value_in

_PI = math.pi
FloatArray: TypeAlias = npt.NDArray[np.float64]


def _zeros(size: int) -> FloatArray:
    return cast(FloatArray, np.zeros(size, dtype=float))


def f1(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)

    f[0] = 1.0
    for i in range(1, n_obj):
        f[0] *= math.sin(float(y[i - 1]) * _PI / 2.0)
    f[0] = zcat_fix_to_01(float(f[0]))

    for j in range(2, n_obj):
        value = 1.0
        for i in range(1, n_obj - j + 1):
            value *= math.sin(float(y[i - 1]) * _PI / 2.0)
        value *= math.cos(float(y[n_obj - j]) * _PI / 2.0)
        f[j - 1] = zcat_fix_to_01(value)

    f[n_obj - 1] = zcat_fix_to_01(1.0 - math.sin(float(y[0]) * _PI / 2.0))
    return f


def f2(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)

    value = 1.0
    for i in range(1, n_obj):
        value *= 1.0 - math.cos(float(y[i - 1]) * _PI / 2.0)
    f[0] = value

    for j in range(2, n_obj):
        value = 1.0
        for i in range(1, n_obj - j + 1):
            value *= 1.0 - math.cos(float(y[i - 1]) * _PI / 2.0)
        value *= 1.0 - math.sin(float(y[n_obj - j]) * _PI / 2.0)
        f[j - 1] = value

    f[n_obj - 1] = 1.0 - math.sin(float(y[0]) * _PI / 2.0)
    return f


def f3(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)

    total = 0.0
    for i in range(1, n_obj):
        total += float(y[i - 1])
    f[0] = total / (n_obj - 1.0)

    for j in range(2, n_obj):
        total = 0.0
        for i in range(1, n_obj - j + 1):
            total += float(y[i - 1])
        total += 1.0 - float(y[n_obj - j])
        f[j - 1] = total / (n_obj - j + 1.0)

    f[n_obj - 1] = 1.0 - float(y[0])
    return f


def f4(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)

    total = 0.0
    for j in range(1, n_obj):
        f[j - 1] = float(y[j - 1])
        total += float(y[j - 1])

    f[n_obj - 1] = 1.0 - total / (n_obj - 1.0)
    return f


def f5(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)

    total = 0.0
    for j in range(1, n_obj):
        f[j - 1] = float(y[j - 1])
        total += 1.0 - float(y[j - 1])

    numer = math.exp(total / (n_obj - 1.0)) ** 8.0 - 1.0
    denom = math.exp(1.0) ** 8.0 - 1.0
    f[n_obj - 1] = numer / denom
    return f


def f6(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)
    k = 40.0
    r = 0.05

    mu = 0.0
    for j in range(1, n_obj):
        f[j - 1] = float(y[j - 1])
        mu += float(y[j - 1])
    mu /= n_obj - 1.0

    numer = (1.0 + math.exp(2.0 * k * mu - k)) ** -1.0 - r * mu - (1.0 + math.exp(k)) ** -1.0 + r
    denom = (1.0 + math.exp(-k)) ** -1.0 - (1.0 + math.exp(k)) ** -1.0 + r
    f[n_obj - 1] = numer / denom
    return f


def f7(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)

    total = 0.0
    for j in range(1, n_obj):
        f[j - 1] = float(y[j - 1])
        total += (0.5 - float(y[j - 1])) ** 5.0

    f[n_obj - 1] = total / (2.0 * (n_obj - 1.0) * (0.5**5.0)) + 0.5
    return f


def f8(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)

    value = 1.0
    for i in range(1, n_obj):
        value *= 1.0 - math.sin(float(y[i - 1]) * _PI / 2.0)
    f[0] = 1.0 - value

    for j in range(2, n_obj):
        value = 1.0
        for i in range(1, n_obj - j + 1):
            value *= 1.0 - math.sin(float(y[i - 1]) * _PI / 2.0)
        value *= 1.0 - math.cos(float(y[n_obj - j]) * _PI / 2.0)
        f[j - 1] = 1.0 - value

    f[n_obj - 1] = math.cos(float(y[0]) * _PI / 2.0)
    return f


def f9(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)

    total = 0.0
    for i in range(1, n_obj):
        total += math.sin(float(y[i - 1]) * _PI / 2.0)
    f[0] = total / (n_obj - 1.0)

    for j in range(2, n_obj):
        total = 0.0
        for i in range(1, n_obj - j + 1):
            total += math.sin(float(y[i - 1]) * _PI / 2.0)
        total += math.cos(float(y[n_obj - j]) * _PI / 2.0)
        f[j - 1] = total / (n_obj - j + 1.0)

    f[n_obj - 1] = math.cos(float(y[0]) * _PI / 2.0)
    return f


def f10(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)
    r = 0.02

    total = 0.0
    for j in range(1, n_obj):
        total += 1.0 - float(y[j - 1])
        f[j - 1] = float(y[j - 1])

    numer = (r**-1.0) - ((total / (n_obj - 1.0) + r) ** -1.0)
    denom = (r**-1.0) - ((1.0 + r) ** -1.0)
    f[n_obj - 1] = numer / denom
    return f


def f11(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)
    k = 4.0

    total = 0.0
    for i in range(1, n_obj):
        total += float(y[i - 1])
    f[0] = total / (n_obj - 1.0)

    for j in range(2, n_obj):
        total = 0.0
        for i in range(1, n_obj - j + 1):
            total += float(y[i - 1])
        total += 1.0 - float(y[n_obj - j])
        f[j - 1] = total / (n_obj - j + 1.0)

    y0 = float(y[0])
    f[n_obj - 1] = (math.cos((2.0 * k - 1.0) * y0 * _PI) + 2.0 * y0 + 4.0 * k * (1.0 - y0) - 1.0) / (4.0 * k)
    return f


def f12(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)
    k = 3.0

    value = 1.0
    for i in range(1, n_obj):
        value *= 1.0 - float(y[i - 1])
    f[0] = 1.0 - value

    for j in range(2, n_obj):
        value = 1.0
        for i in range(1, n_obj - j + 1):
            value *= 1.0 - float(y[i - 1])
        value *= float(y[n_obj - j])
        f[j - 1] = 1.0 - value

    y0 = float(y[0])
    f[n_obj - 1] = (math.cos((2.0 * k - 1.0) * y0 * _PI) + 2.0 * y0 + 4.0 * k * (1.0 - y0) - 1.0) / (4.0 * k)
    return f


def f13(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)
    k = 3.0

    total = 0.0
    for i in range(1, n_obj):
        total += math.sin(float(y[i - 1]) * _PI / 2.0)
    f[0] = 1.0 - total / (n_obj - 1.0)

    for j in range(2, n_obj):
        total = 0.0
        for i in range(1, n_obj - j + 1):
            total += math.sin(float(y[i - 1]) * _PI / 2.0)
        total += math.cos(float(y[n_obj - j]) * _PI / 2.0)
        f[j - 1] = 1.0 - total / (n_obj - j + 1.0)

    y0 = float(y[0])
    f[n_obj - 1] = 1.0 - (
        math.cos((2.0 * k - 1.0) * y0 * _PI) + 2.0 * y0 + 4.0 * k * (1.0 - y0) - 1.0
    ) / (4.0 * k)
    return f


def f14(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)
    y0 = float(y[0])
    sin_term = math.sin(y0 * _PI / 2.0)

    f[0] = sin_term**2.0
    for j in range(2, n_obj - 1):
        f[j - 1] = sin_term ** (2.0 + (j - 1.0) / (n_obj - 2.0))

    if n_obj > 2:
        f[n_obj - 2] = 0.5 * (1.0 + math.sin(6.0 * y0 * _PI / 2.0 - _PI / 2.0))

    f[n_obj - 1] = math.cos(y0 * _PI / 2.0)
    return f


def f15(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)
    k = 3.0
    y0 = float(y[0])

    for j in range(1, n_obj):
        f[j - 1] = y0 ** (1.0 + (j - 1.0) / (4.0 * n_obj))

    f[n_obj - 1] = (math.cos((2.0 * k - 1.0) * y0 * _PI) + 2.0 * y0 + 4.0 * k * (1.0 - y0) - 1.0) / (4.0 * k)
    return f


def f16(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)
    k = 5.0
    y0 = float(y[0])
    sin_term = math.sin(y0 * _PI / 2.0)

    f[0] = sin_term
    for j in range(2, n_obj - 1):
        f[j - 1] = sin_term ** (1.0 + (j - 1.0) / (n_obj - 2.0))

    if n_obj > 2:
        f[n_obj - 2] = 0.5 * (1.0 + math.sin(10.0 * y0 * _PI / 2.0 - _PI / 2.0))

    f[n_obj - 1] = (math.cos((2.0 * k - 1.0) * y0 * _PI) + 2.0 * y0 + 4.0 * k * (1.0 - y0) - 1.0) / (4.0 * k)
    return f


def f17(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)

    wedge_flag = True
    for j in range(0, n_obj - 1):
        yj = float(y[j])
        if yj < 0.0 or yj > 0.5:
            wedge_flag = False
            break

    total = 0.0
    for j in range(1, n_obj):
        if wedge_flag:
            f[j - 1] = float(y[0])
        else:
            f[j - 1] = float(y[j - 1])
            total += 1.0 - float(y[j - 1])

    if wedge_flag:
        numer = math.exp(1.0 - float(y[0])) ** 8.0 - 1.0
    else:
        numer = math.exp(total / (n_obj - 1.0)) ** 8.0 - 1.0
    denom = math.exp(1.0) ** 8.0 - 1.0
    f[n_obj - 1] = numer / denom
    return f


def f18(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)

    in_first = zcat_for_all_value_in(y, n_obj - 1, 0.0, 0.4)
    in_second = zcat_for_all_value_in(y, n_obj - 1, 0.6, 1.0)
    wedge_flag = in_first or in_second

    total = 0.0
    for j in range(1, n_obj):
        if wedge_flag:
            f[j - 1] = float(y[0])
        else:
            f[j - 1] = float(y[j - 1])
            total += (0.5 - float(y[j - 1])) ** 5.0

    if wedge_flag:
        y0 = float(y[0])
        f[n_obj - 1] = ((0.5 - y0) ** 5.0 + 0.5**5.0) / (2.0 * (0.5**5.0))
    else:
        f[n_obj - 1] = total / (2.0 * (n_obj - 1.0) * (0.5**5.0)) + 0.5
    return f


def f19(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)
    a = 5.0

    flag_deg = zcat_value_in(float(y[0]), 0.0, 0.2) or zcat_value_in(float(y[0]), 0.4, 0.6)

    mu = 0.0
    for j in range(1, n_obj):
        mu += float(y[j - 1])
        f[j - 1] = float(y[0]) if flag_deg else float(y[j - 1])
        f[j - 1] = zcat_fix_to_01(float(f[j - 1]))
    mu = float(y[0]) if flag_deg else mu / (n_obj - 1.0)

    f[n_obj - 1] = 1.0 - mu - math.cos(2.0 * a * _PI * mu + _PI / 2.0) / (2.0 * a * _PI)
    f[n_obj - 1] = zcat_fix_to_01(float(f[n_obj - 1]))
    return f


def f20(y: FloatArray, n_obj: int) -> FloatArray:
    f = _zeros(n_obj)

    deg_flag = zcat_value_in(float(y[0]), 0.1, 0.4) or zcat_value_in(float(y[0]), 0.6, 0.9)

    total = 0.0
    for j in range(1, n_obj):
        total += (0.5 - float(y[j - 1])) ** 5.0
        f[j - 1] = float(y[0]) if deg_flag else float(y[j - 1])

    if deg_flag:
        y0 = float(y[0])
        f[n_obj - 1] = ((0.5 - y0) ** 5.0 + 0.5**5.0) / (2.0 * (0.5**5.0))
    else:
        f[n_obj - 1] = total / (2.0 * (n_obj - 1.0) * (0.5**5.0)) + 0.5
    return f


def evaluate_f(function_id: int, y: FloatArray, n_obj: int) -> FloatArray:
    if function_id == 1:
        return f1(y, n_obj)
    if function_id == 2:
        return f2(y, n_obj)
    if function_id == 3:
        return f3(y, n_obj)
    if function_id == 4:
        return f4(y, n_obj)
    if function_id == 5:
        return f5(y, n_obj)
    if function_id == 6:
        return f6(y, n_obj)
    if function_id == 7:
        return f7(y, n_obj)
    if function_id == 8:
        return f8(y, n_obj)
    if function_id == 9:
        return f9(y, n_obj)
    if function_id == 10:
        return f10(y, n_obj)
    if function_id == 11:
        return f11(y, n_obj)
    if function_id == 12:
        return f12(y, n_obj)
    if function_id == 13:
        return f13(y, n_obj)
    if function_id == 14:
        return f14(y, n_obj)
    if function_id == 15:
        return f15(y, n_obj)
    if function_id == 16:
        return f16(y, n_obj)
    if function_id == 17:
        return f17(y, n_obj)
    if function_id == 18:
        return f18(y, n_obj)
    if function_id == 19:
        return f19(y, n_obj)
    if function_id == 20:
        return f20(y, n_obj)
    raise ValueError(f"Unsupported F-function id: {function_id}")


__all__ = ["evaluate_f"]
