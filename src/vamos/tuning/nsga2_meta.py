from __future__ import annotations

import math
import time
from typing import List, Tuple

import numpy as np

from vamos.tuning.meta_problem import MetaOptimizationProblem


class MetaNSGAII:
    """
    Minimal NSGA-II loop operating on the meta-configuration space.
    """

    def __init__(
        self,
        problem: MetaOptimizationProblem,
        population_size: int = 50,
        offspring_size: int | None = None,
        max_meta_evals: int = 2000,
        p_crossover: float = 0.9,
        p_mutation: float | None = None,
        eta_c: float = 20.0,
        eta_m: float = 20.0,
        seed: int | None = None,
        max_total_inner_runs: int | None = None,
        max_wall_time: float | None = None,
    ):
        self.problem = problem
        self.dim = problem.config_space.dim()
        self.population_size = int(population_size)
        if self.population_size <= 0:
            raise ValueError("population_size must be positive.")
        self.offspring_size = int(offspring_size or population_size)
        if self.offspring_size <= 0:
            raise ValueError("offspring_size must be positive.")
        self.max_meta_evals = int(max_meta_evals)
        if self.max_meta_evals <= 0:
            raise ValueError("max_meta_evals must be positive.")
        if self.max_meta_evals < self.population_size:
            raise ValueError("max_meta_evals must be at least the population size.")
        self.p_crossover = float(p_crossover)
        self.p_mutation = float(p_mutation) if p_mutation is not None else 1.0 / max(1, self.dim)
        self.eta_c = float(eta_c)
        self.eta_m = float(eta_m)
        self.rng = np.random.default_rng(seed)
        self.max_total_inner_runs = max_total_inner_runs
        self.max_wall_time = max_wall_time
        self.meta_eval_count = 0
        self.inner_run_count = 0
        self.start_time = None
        self._seen_cache_keys: set[str] = set()

    def run(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        self.start_time = time.perf_counter()
        X = self.rng.uniform(0.0, 1.0, size=(self.population_size, self.dim))
        F, evaluated = self._evaluate_population(X)
        X = X[:evaluated]
        if evaluated == 0:
            diagnostics = self._diagnostics()
            return np.empty((0, self.dim)), np.empty((0, 0)), diagnostics

        while not self._budget_exhausted():
            remaining_meta = self.max_meta_evals - self.meta_eval_count
            if remaining_meta <= 0:
                break
            remaining = min(self.offspring_size, remaining_meta)
            if self.max_total_inner_runs is not None:
                remaining = min(remaining, self.max_total_inner_runs - self.inner_run_count)
            if remaining <= 0:
                break
            offspring = self._create_offspring(X, F, remaining)
            off_F, evaluated_off = self._evaluate_population(offspring)
            offspring = offspring[:evaluated_off]
            if evaluated_off == 0:
                break
            X, F = self._survival(X, F, offspring, off_F)

        final_fronts = self._non_dominated_sort(F)
        final_front = final_fronts[0] if final_fronts else []
        diagnostics = self._diagnostics()
        return X[final_front], F[final_front], diagnostics

    def _evaluate_population(self, X: np.ndarray) -> Tuple[np.ndarray, int]:
        objs: List[np.ndarray] = []
        evaluated = 0
        for vec in X:
            if self._budget_exhausted():
                break
            cache_key = self.problem.cache_key_for_vector(vec)
            cached = self.problem.cached_objectives(cache_key)
            if cached is not None:
                value_arr = np.asarray(cached, dtype=float).ravel()
            else:
                value = self.problem.evaluate(vec)
                value_arr = np.asarray(value, dtype=float).ravel()
                self.meta_eval_count += 1
                self.inner_run_count += getattr(self.problem, "last_inner_runs", 0)
            if value_arr.size == 0:
                raise ValueError("Meta problem returned an empty objective array.")
            objs.append(value_arr)
            evaluated += 1
            self._seen_cache_keys.add(cache_key)
        if not objs:
            return np.empty((0, 0), dtype=float), 0
        F = np.vstack(objs)
        return F, evaluated

    def _create_offspring(self, X: np.ndarray, F: np.ndarray, batch_size: int) -> np.ndarray:
        ranks, crowding = self._rank_and_crowding(F)
        pairs_needed = math.ceil(batch_size / 2)
        offspring: List[np.ndarray] = []
        for _ in range(pairs_needed):
            p1 = self._tournament_select(ranks, crowding)
            p2 = self._tournament_select(ranks, crowding)
            c1, c2 = self._crossover(X[p1], X[p2])
            offspring.append(self._mutate(c1))
            if len(offspring) < batch_size:
                offspring.append(self._mutate(c2))
        stacked = np.vstack(offspring)
        return stacked[:batch_size]

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.p_crossover:
            return p1.copy(), p2.copy()
        u = self.rng.random(self.dim)
        beta = np.empty_like(u)
        mask = u <= 0.5
        beta[mask] = (2.0 * u[mask]) ** (1.0 / (self.eta_c + 1.0))
        beta[~mask] = (1.0 / (2.0 * (1.0 - u[~mask]))) ** (1.0 / (self.eta_c + 1.0))
        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        return np.clip(c1, 0.0, 1.0), np.clip(c2, 0.0, 1.0)

    def _mutate(self, child: np.ndarray) -> np.ndarray:
        mask = self.rng.random(self.dim) < self.p_mutation
        if not mask.any():
            return child
        u = self.rng.random(self.dim)
        delta = np.empty(self.dim)
        lower = child
        upper = 1.0 - child
        for i, active in enumerate(mask):
            if not active:
                delta[i] = 0.0
                continue
            if u[i] < 0.5:
                delta_q = (2.0 * u[i]) ** (1.0 / (self.eta_m + 1.0)) - 1.0
                delta[i] = delta_q * lower[i]
            else:
                delta_q = 1.0 - (2.0 * (1.0 - u[i])) ** (1.0 / (self.eta_m + 1.0))
                delta[i] = delta_q * upper[i]
        mutated = child + delta
        return np.clip(mutated, 0.0, 1.0)

    def _rank_and_crowding(self, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        fronts = self._non_dominated_sort(F)
        ranks = np.empty(F.shape[0], dtype=int)
        crowding = np.zeros(F.shape[0], dtype=float)
        for rank, front in enumerate(fronts):
            ranks[front] = rank
            crowding[front] = self._crowding_distance(F, front)
        return ranks, crowding

    def _tournament_select(self, ranks: np.ndarray, crowding: np.ndarray) -> int:
        idx_a, idx_b = self.rng.integers(0, ranks.size, size=2)
        rank_a, rank_b = ranks[idx_a], ranks[idx_b]
        if rank_a < rank_b:
            return idx_a
        if rank_b < rank_a:
            return idx_b
        if crowding[idx_a] > crowding[idx_b]:
            return idx_a
        if crowding[idx_b] > crowding[idx_a]:
            return idx_b
        return idx_a if self.rng.random() < 0.5 else idx_b

    def _survival(
        self,
        X: np.ndarray,
        F: np.ndarray,
        offspring: np.ndarray,
        off_F: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        combined_X = np.vstack((X, offspring))
        combined_F = np.vstack((F, off_F))
        fronts = self._non_dominated_sort(combined_F)
        survivors: List[int] = []
        for front in fronts:
            if len(survivors) + len(front) <= self.population_size:
                survivors.extend(front)
            else:
                remaining = self.population_size - len(survivors)
                if remaining <= 0:
                    break
                cd = self._crowding_distance(combined_F, front)
                order = np.argsort(-cd)
                selected = [front[idx] for idx in order[:remaining]]
                survivors.extend(selected)
                break
        survivors_array = np.array(survivors, dtype=int)
        return combined_X[survivors_array], combined_F[survivors_array]

    @staticmethod
    def _non_dominated_sort(F: np.ndarray) -> List[List[int]]:
        n_points = F.shape[0]
        domination_counts = np.zeros(n_points, dtype=int)
        dominates: List[List[int]] = [[] for _ in range(n_points)]

        for i in range(n_points):
            for j in range(i + 1, n_points):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominates[i].append(j)
                    domination_counts[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominates[j].append(i)
                    domination_counts[i] += 1

        fronts: List[List[int]] = []
        current = [i for i in range(n_points) if domination_counts[i] == 0]
        while current:
            fronts.append(current)
            next_front: List[int] = []
            for p in current:
                for q in dominates[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        next_front.append(q)
            current = next_front
        if not fronts:
            fronts.append(list(range(n_points)))
        return fronts

    @staticmethod
    def _crowding_distance(F: np.ndarray, front: List[int]) -> np.ndarray:
        if not front:
            return np.array([], dtype=float)
        m = F.shape[1]
        distances = np.zeros(len(front), dtype=float)
        front_array = np.asarray(front, dtype=int)
        for obj in range(m):
            values = F[front_array, obj]
            order = np.argsort(values)
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf
            min_val = values[order[0]]
            max_val = values[order[-1]]
            if max_val - min_val <= 0.0:
                continue
            for idx in range(1, len(front) - 1):
                prev_idx = order[idx - 1]
                next_idx = order[idx + 1]
                distances[order[idx]] += (values[next_idx] - values[prev_idx]) / (max_val - min_val)
        return distances

    def _budget_exhausted(self) -> bool:
        if self.meta_eval_count >= self.max_meta_evals:
            return True
        if self.max_total_inner_runs is not None and self.inner_run_count >= self.max_total_inner_runs:
            return True
        if self.max_wall_time is not None and self.start_time is not None:
            if (time.perf_counter() - self.start_time) >= self.max_wall_time:
                return True
        return False

    def _diagnostics(self) -> dict:
        elapsed = 0.0
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
        return {
            "n_meta_evals": self.meta_eval_count,
            "n_inner_runs": self.inner_run_count,
            "elapsed_time": elapsed,
        }
