import numpy as np

from vamos.engine.operators.binary import (
    random_binary_population,
    one_point_crossover,
    two_point_crossover,
    uniform_crossover,
    bit_flip_mutation,
)
from vamos.engine.operators.integer import (
    random_integer_population,
    uniform_integer_crossover,
    arithmetic_integer_crossover,
    random_reset_mutation,
    creep_mutation,
)
from vamos.engine.operators.real import SBXCrossover, PolynomialMutation, VariationWorkspace

from .weight_vectors import load_or_generate_weight_vectors


def _resolve_prob_expression(value, n_var: int, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, str) and value.endswith("/n"):
        numerator = value[:-2]
        num = float(numerator) if numerator else 1.0
        return min(1.0, max(num, 0.0) / n_var)
    return float(value)


_BINARY_CROSSOVER = {
    "one_point": one_point_crossover,
    "single_point": one_point_crossover,
    "1point": one_point_crossover,
    "two_point": two_point_crossover,
    "2point": two_point_crossover,
    "uniform": uniform_crossover,
}

_BINARY_MUTATION = {
    "bitflip": bit_flip_mutation,
    "bit_flip": bit_flip_mutation,
}

_INT_CROSSOVER = {
    "uniform": uniform_integer_crossover,
    "blend": arithmetic_integer_crossover,
    "arithmetic": arithmetic_integer_crossover,
}

_INT_MUTATION = {
    "reset": random_reset_mutation,
    "random_reset": random_reset_mutation,
    "creep": creep_mutation,
}


class NSGAIII:
    """
    Simplified NSGA-III implementation that reuses the existing vectorized kernels
    for variation while performing NSGA-III niching during survival. Supports
    binary/integer encodings with dedicated operators.
    """

    def __init__(self, config: dict, kernel):
        self.cfg = config
        self.kernel = kernel

    def run(self, problem, termination, seed: int):
        term_type, term_val = termination
        assert term_type == "n_eval", "Only termination=('n_eval', N) is supported."
        max_eval = term_val

        rng = np.random.default_rng(seed)
        pop_size = self.cfg["pop_size"]
        encoding = getattr(problem, "encoding", "continuous")
        bounds_dtype = int if encoding == "integer" else float
        xl = np.asarray(problem.xl, dtype=bounds_dtype)
        xu = np.asarray(problem.xu, dtype=bounds_dtype)
        n_var = problem.n_var
        n_obj = problem.n_obj
        if xl.ndim == 0:
            xl = np.full(n_var, xl, dtype=bounds_dtype)
        if xu.ndim == 0:
            xu = np.full(n_var, xu, dtype=bounds_dtype)

        cross_method, cross_params = self.cfg["crossover"]
        cross_params = dict(cross_params)

        mut_method, mut_params = self.cfg["mutation"]
        mut_params = dict(mut_params)
        if mut_params.get("prob") == "1/n":
            mut_params["prob"] = 1.0 / n_var

        if encoding == "binary":
            if cross_method not in _BINARY_CROSSOVER:
                raise ValueError(f"Unsupported NSGA-III crossover '{cross_method}' for binary encoding.")
            if mut_method not in _BINARY_MUTATION:
                raise ValueError(f"Unsupported NSGA-III mutation '{mut_method}' for binary encoding.")
            cross_fn = _BINARY_CROSSOVER[cross_method]
            cross_prob = float(cross_params.get("prob", 0.9))
            mut_fn = _BINARY_MUTATION[mut_method]
            mut_prob = _resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
            crossover = lambda parents: cross_fn(parents, cross_prob, rng)
            mutation = lambda X_child: (mut_fn(X_child, mut_prob, rng) or X_child)
        elif encoding == "integer":
            if cross_method not in _INT_CROSSOVER:
                raise ValueError(f"Unsupported NSGA-III crossover '{cross_method}' for integer encoding.")
            if mut_method not in _INT_MUTATION:
                raise ValueError(f"Unsupported NSGA-III mutation '{mut_method}' for integer encoding.")
            cross_fn = _INT_CROSSOVER[cross_method]
            cross_prob = float(cross_params.get("prob", 0.9))
            mut_fn = _INT_MUTATION[mut_method]
            mut_prob = _resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
            step = int(mut_params.get("step", 1))
            crossover = lambda parents: cross_fn(parents, cross_prob, rng)
            mutation = (
                (lambda X_child: (mut_fn(X_child, mut_prob, step, xl, xu, rng) or X_child))
                if mut_fn is creep_mutation
                else (lambda X_child: (mut_fn(X_child, mut_prob, xl, xu, rng) or X_child))
            )
        elif encoding in {"continuous", "real"}:
            cross_prob = float(cross_params.get("prob", 0.9))
            cross_eta = float(cross_params.get("eta", 20.0))
            workspace = VariationWorkspace()
            sbx = SBXCrossover(
                prob_crossover=cross_prob,
                eta=cross_eta,
                lower=xl,
                upper=xu,
                workspace=workspace,
                allow_inplace=True,
            )
            mut_prob = _resolve_prob_expression(mut_params.get("prob"), n_var, 1.0 / max(1, n_var))
            mut_eta = float(mut_params.get("eta", 20.0))
            pm = PolynomialMutation(
                prob_mutation=mut_prob,
                eta=mut_eta,
                lower=xl,
                upper=xu,
                workspace=workspace,
            )
            crossover = lambda parents: sbx(parents, rng)
            mutation = lambda X_child: pm(X_child, rng)
        else:
            raise ValueError(f"NSGA-III does not support encoding '{encoding}'.")

        sel_method, sel_params = self.cfg["selection"]
        assert sel_method == "tournament"
        pressure = sel_params.get("pressure", 2)

        dir_cfg = self.cfg.get("reference_directions", {}) or {}
        ref_dirs = load_or_generate_weight_vectors(
            pop_size, n_obj, path=dir_cfg.get("path"), divisions=dir_cfg.get("divisions")
        )
        if ref_dirs.shape[0] > pop_size:
            ref_dirs = ref_dirs[:pop_size]

        ref_dirs = np.asarray(ref_dirs, dtype=float)
        ref_dirs_norm = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
        ref_dirs_norm[np.isnan(ref_dirs_norm)] = 0.0

        if encoding == "binary":
            X = random_binary_population(pop_size, n_var, rng)
        elif encoding == "integer":
            X = random_integer_population(pop_size, n_var, xl.astype(int), xu.astype(int), rng)
        else:
            X = rng.uniform(xl, xu, size=(pop_size, n_var))
        F = np.empty((pop_size, n_obj))
        problem.evaluate(X, {"F": F})
        n_eval = pop_size

        while n_eval < max_eval:
            ranks, crowd = self.kernel.nsga2_ranking(F)
            parents_idx = self.kernel.tournament_selection(
                ranks, crowd, pressure, rng, n_parents=2 * (pop_size // 2)
            )
            X_parents = X[parents_idx].reshape(-1, 2, n_var)
            offspring_pairs = crossover(X_parents)
            X_off = offspring_pairs.reshape(-1, n_var)
            X_off = mutation(X_off)

            F_off = np.empty((X_off.shape[0], n_obj))
            problem.evaluate(X_off, {"F": F_off})
            n_eval += X_off.shape[0]

            X = np.vstack([X, X_off])
            F = np.vstack([F, F_off])
            X, F = self._nsga3_survival(X, F, pop_size, ref_dirs_norm, rng)

        return {"X": X, "F": F, "reference_directions": ref_dirs}

    def _nsga3_survival(self, X, F, pop_size, ref_dirs_norm, rng):
        fronts = self._fast_non_dominated_sort(F)
        new_X = []
        new_F = []

        ideal = F.min(axis=0)
        shifted = F - ideal
        extreme_idx = self._identify_extremes(shifted)
        intercepts = self._compute_intercepts(shifted, extreme_idx)
        denom = np.where(intercepts > 0, intercepts, 1.0)
        normalized = shifted / denom

        associations, distances = self._associate(normalized, ref_dirs_norm)
        niche_counts = np.zeros(ref_dirs_norm.shape[0], dtype=int)

        for front in fronts:
            front = np.asarray(front, dtype=int)
            if len(new_X) + front.size <= pop_size:
                new_X.extend(X[front])
                new_F.extend(F[front])
                for idx in front:
                    niche_counts[associations[idx]] += 1
            else:
                remaining = pop_size - len(new_X)
                selected_idx = self._niche_selection(
                    front, remaining, niche_counts, associations, distances, rng
                )
                new_X.extend(X[selected_idx])
                new_F.extend(F[selected_idx])
                break

        return np.asarray(new_X), np.asarray(new_F)

    @staticmethod
    def _identify_extremes(shifted: np.ndarray) -> np.ndarray:
        """Identify extreme points using ASF (Achievement Scalarization Function)."""
        if shifted.size == 0:
            return np.array([], dtype=int)
        n_obj = shifted.shape[1]
        extremes = np.empty(n_obj, dtype=int)
        unit = np.eye(n_obj)
        for i in range(n_obj):
            weights = np.where(unit[i] == 0, 1e6, 1.0)
            asf = (shifted * weights).max(axis=1)
            extremes[i] = int(np.argmin(asf))
        return extremes

    @staticmethod
    def _compute_intercepts(shifted: np.ndarray, extreme_idx: np.ndarray) -> np.ndarray:
        """Compute intercepts from extreme points; fall back to axis-wise maxima."""
        n_obj = shifted.shape[1]
        if extreme_idx.size == 0:
            return np.ones(n_obj, dtype=float)
        extreme_pts = shifted[extreme_idx]
        intercepts = np.zeros(n_obj, dtype=float)
        try:
            b = np.ones(n_obj)
            plane = np.linalg.solve(extreme_pts, b)
            intercepts = 1.0 / plane
        except Exception:
            intercepts = shifted.max(axis=0)
        if np.any(~np.isfinite(intercepts)) or np.any(intercepts <= 1e-12):
            intercepts = shifted.max(axis=0)
        intercepts = np.where(intercepts > 0, intercepts, 1.0)
        return intercepts

    @staticmethod
    def _associate(normalized_F, ref_dirs_norm):
        norms = np.linalg.norm(normalized_F, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1e-12)
        normalized_vectors = normalized_F / norms
        cosine = normalized_vectors @ ref_dirs_norm.T
        cosine = np.clip(cosine, -1.0, 1.0)
        associations = np.argmax(cosine, axis=1)
        cos_selected = cosine[np.arange(cosine.shape[0]), associations]
        distances = norms.flatten() * np.sqrt(1.0 - np.square(cos_selected))
        return associations, distances

    def _niche_selection(
        self, front, n_remaining, niche_counts, associations, distances, rng
    ):
        selected = []
        pool = front.tolist()
        while len(selected) < n_remaining and pool:
            assoc_front = np.array([associations[idx] for idx in pool])
            unique_refs = np.unique(assoc_front)
            ref_counts = niche_counts[unique_refs]
            min_count = np.min(ref_counts)
            candidate_refs = unique_refs[ref_counts == min_count]
            ref_choice = rng.choice(candidate_refs)

            candidates = [idx for idx in pool if associations[idx] == ref_choice]
            if not candidates:
                niche_counts[ref_choice] = np.inf
                continue
            cand_dist = np.array([distances[idx] for idx in candidates])
            best = candidates[np.argmin(cand_dist)]
            pool.remove(best)
            niche_counts[ref_choice] += 1
            selected.append(best)
        if len(selected) < n_remaining and pool:
            selected.extend(rng.choice(pool, size=n_remaining - len(selected), replace=False))
        return np.asarray(selected, dtype=int)

    @staticmethod
    def _fast_non_dominated_sort(F):
        n = F.shape[0]
        S = [[] for _ in range(n)]
        domination_count = np.zeros(n, dtype=int)
        ranks = np.zeros(n, dtype=int)
        fronts = [[]]

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    domination_count[p] += 1
            if domination_count[p] == 0:
                ranks[p] = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        ranks[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        fronts.pop()  # remove last empty front
        return fronts
