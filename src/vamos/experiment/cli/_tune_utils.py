from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable
from typing import Any

import numpy as np

from vamos.engine.tuning import Instance


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def parse_csv_ints(raw: str | None, parser: argparse.ArgumentParser, flag: str, *, min_len: int = 1) -> tuple[int, ...] | None:
    if raw is None:
        return None
    parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if len(parts) < min_len:
        parser.error(f"{flag} must provide at least {min_len} comma-separated integers.")
    try:
        values = tuple(int(part) for part in parts)
    except ValueError:
        parser.error(f"{flag} must be comma-separated integers.")
    if any(v <= 0 for v in values):
        parser.error(f"{flag} values must be > 0.")
    return values


def parse_csv_strings(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(chunk.strip() for chunk in raw.split(",") if chunk.strip())


def parse_seed_spec(raw: str | None, *, default_start: int, default_count: int) -> list[int]:
    if raw is None or not raw.strip():
        return [default_start + i for i in range(default_count)]
    out: list[int] = []
    for token in parse_csv_strings(raw):
        if ":" in token:
            parts = token.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid seed range token: {token!r}")
            lo = int(parts[0].strip())
            hi = int(parts[1].strip())
            if hi <= lo:
                raise ValueError(f"Invalid seed range {token!r}: end must be > start.")
            out.extend(list(range(lo, hi)))
        else:
            out.append(int(token))
    if not out:
        raise ValueError("Seed specification resolved to empty list.")
    return out


def resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs == -1:
        return max(1, int(os.cpu_count() or 1) - 1)
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1 or -1.")
    return int(n_jobs)


def parse_ref_point(raw: str | None, n_obj: int) -> list[float]:
    if raw:
        try:
            parsed = [float(x.strip()) for x in raw.split(",")]
            if len(parsed) == n_obj:
                return parsed
            _logger().warning("Reference point length (%s) does not match n_obj=%s. Falling back to default.", len(parsed), n_obj)
        except ValueError:
            _logger().warning("Failed to parse --ref-point. Falling back to default.")
    return [10.0] * n_obj


def build_aggregator(mode: str) -> Callable[[list[float]], float]:
    m = str(mode).strip().lower()
    if m == "mean":
        return lambda scores: float(np.mean(scores))
    if m == "median":
        return lambda scores: float(np.median(scores))
    if m == "p25":
        return lambda scores: float(np.percentile(scores, 25))
    if m == "p10":
        return lambda scores: float(np.percentile(scores, 10))
    raise ValueError(f"Unsupported aggregate mode: {mode!r}")


def infer_suite(instance_name: str) -> str:
    lower = str(instance_name).strip().lower()
    if lower.startswith("zdt"):
        return "zdt"
    if lower.startswith("dtlz"):
        return "dtlz"
    if lower.startswith("wfg"):
        return "wfg"
    if lower.startswith("uf"):
        return "uf"
    if lower.startswith("cf"):
        return "cf"
    if lower.startswith("re"):
        return "re"
    if lower.startswith("mw"):
        return "mw"
    return "other"


def split_counts(n: int, train_frac: float, validation_frac: float) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    n_train = max(1, int(round(n * train_frac)))
    n_valid = max(1, int(round(n * validation_frac)))
    if n_train + n_valid >= n:
        n_valid = max(1, n - n_train - 1)
    n_test = max(1, n - n_train - n_valid)
    while n_train + n_valid + n_test > n:
        if n_train > n_valid and n_train > 1:
            n_train -= 1
        elif n_valid > 1:
            n_valid -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    return int(n_train), int(n_valid), int(n_test)


def split_instances(
    instances: list[Instance],
    *,
    train_frac: float,
    validation_frac: float,
    split_seed: int,
    strategy: str,
) -> tuple[list[Instance], list[Instance], list[Instance], list[dict[str, Any]]]:
    n = len(instances)
    if n == 0:
        raise ValueError("No instances provided for tuning.")
    rng = np.random.default_rng(int(split_seed))
    perm = np.arange(n, dtype=int)
    rng.shuffle(perm)
    ordered = [instances[i] for i in perm]

    if n == 1:
        only = [ordered[0]]
        manifest = [
            {
                "instance": ordered[0].name,
                "suite": infer_suite(ordered[0].name),
                "split": "train/validation/test",
                "shared_instance": True,
            }
        ]
        _logger().warning("Only one instance provided; train/validation/test share instance but use disjoint seeds.")
        return only, only, only, manifest
    if n == 2:
        train = [ordered[0]]
        validation = [ordered[1]]
        test = [ordered[1]]
        manifest = [
            {"instance": ordered[0].name, "suite": infer_suite(ordered[0].name), "split": "train", "shared_instance": False},
            {"instance": ordered[1].name, "suite": infer_suite(ordered[1].name), "split": "validation/test", "shared_instance": True},
        ]
        _logger().warning("Only two instances provided; validation and test share one instance but use disjoint seeds.")
        return train, validation, test, manifest

    if str(strategy) == "suite_stratified":
        suite_groups: dict[str, list[Instance]] = {}
        for inst in ordered:
            suite_groups.setdefault(infer_suite(inst.name), []).append(inst)
        train = []
        validation = []
        test = []
        for suite_name, group in sorted(suite_groups.items()):
            perm_local = np.arange(len(group), dtype=int)
            rng.shuffle(perm_local)
            g = [group[i] for i in perm_local]
            g_train, g_valid, g_test = split_counts(len(g), train_frac, validation_frac)
            train.extend(g[:g_train])
            validation.extend(g[g_train : g_train + g_valid])
            test.extend(g[g_train + g_valid : g_train + g_valid + g_test])
            _logger().debug(
                "Suite split %s -> train=%s validation=%s test=%s",
                suite_name,
                g_train,
                g_valid,
                g_test,
            )
        if not validation and len(train) > 1:
            validation.append(train.pop())
        if not test and len(train) > 1:
            test.append(train.pop())
    else:
        n_train, n_valid, n_test = split_counts(n, train_frac, validation_frac)
        train = ordered[:n_train]
        validation = ordered[n_train : n_train + n_valid]
        test = ordered[n_train + n_valid : n_train + n_valid + n_test]

    manifest = (
        [{"instance": inst.name, "suite": infer_suite(inst.name), "split": "train", "shared_instance": False} for inst in train]
        + [{"instance": inst.name, "suite": infer_suite(inst.name), "split": "validation", "shared_instance": False} for inst in validation]
        + [{"instance": inst.name, "suite": infer_suite(inst.name), "split": "test", "shared_instance": False} for inst in test]
    )
    return train, validation, test, manifest


def resolve_split_seeds(args: argparse.Namespace) -> tuple[list[int], list[int], list[int]]:
    train = parse_seed_spec(None, default_start=int(args.seed), default_count=int(args.n_seeds))
    validation = parse_seed_spec(
        args.validation_seeds,
        default_start=int(args.seed) + 10_000,
        default_count=int(args.n_seeds),
    )
    test = parse_seed_spec(
        args.test_seeds,
        default_start=int(args.seed) + 20_000,
        default_count=int(args.n_seeds),
    )
    st = set(train)
    sv = set(validation)
    ss = set(test)
    if st & sv or st & ss or sv & ss:
        raise ValueError("Seed splits must be disjoint across train/validation/test.")
    return train, validation, test
