import numpy as np

from .weight_vectors import load_or_generate_weight_vectors


class NSGAIII:
    """
    Simplified NSGA-III implementation that reuses the existing vectorized kernels
    for variation while performing NSGA-III niching during survival.
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
        xl, xu = problem.xl, problem.xu
        n_var = problem.n_var
        n_obj = problem.n_obj

        cross_method, cross_params = self.cfg["crossover"]
        assert cross_method == "sbx"
        cross_params = dict(cross_params)

        mut_method, mut_params = self.cfg["mutation"]
        assert mut_method == "pm"
        mut_params = dict(mut_params)
        if mut_params.get("prob") == "1/n":
            mut_params["prob"] = 1.0 / n_var

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

        X = rng.uniform(xl, xu, size=(pop_size, n_var))
        F = np.empty((pop_size, n_obj))
        problem.evaluate(X, {"F": F})
        n_eval = pop_size

        while n_eval < max_eval:
            ranks, crowd = self.kernel.nsga2_ranking(F)
            parents_idx = self.kernel.tournament_selection(
                ranks, crowd, pressure, rng, n_parents=2 * (pop_size // 2)
            )
            X_parents = X[parents_idx]
            X_off = self.kernel.sbx_crossover(X_parents, cross_params, rng, xl, xu)
            self.kernel.polynomial_mutation(X_off, mut_params, rng, xl, xu)

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
        denominator = np.max(shifted, axis=0)
        denominator = np.where(denominator > 0, denominator, 1.0)
        normalized = shifted / denominator

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
