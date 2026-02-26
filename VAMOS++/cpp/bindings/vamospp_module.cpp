#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

using NDArray2dDConst = nb::ndarray<const double, nb::ndim<2>, nb::c_contig>;
using NDArray1dDConst = nb::ndarray<const double, nb::ndim<1>, nb::c_contig>;
using NDArray1dIConst = nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig>;
using NDArray2dD = nb::ndarray<double, nb::ndim<2>, nb::c_contig>;
using NDArray1dI = nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig>;
using Fronts = std::vector<std::vector<size_t>>;

namespace {

inline nb::object np_module() {
    return nb::module_::import_("numpy");
}

inline nb::object make_float64_array_2d(nb::handle np, const std::vector<double>& flat, size_t rows, size_t cols) {
    if (flat.size() != rows * cols) {
        throw std::runtime_error("internal error: flat buffer size mismatch for float64 array.");
    }
    nb::object arr_obj = np.attr("empty")(nb::make_tuple(rows, cols), "float64");
    auto arr = nb::cast<NDArray2dD>(arr_obj);
    std::copy(flat.begin(), flat.end(), arr.data());
    return arr_obj;
}

inline nb::object make_int64_array_1d(nb::handle np, const std::vector<int64_t>& values) {
    nb::object arr_obj = np.attr("empty")(values.size(), "int64");
    auto arr = nb::cast<NDArray1dI>(arr_obj);
    std::copy(values.begin(), values.end(), arr.data());
    return arr_obj;
}

inline NDArray2dD require_out_float64_c_2d(nb::handle out_obj, size_t rows, size_t cols, const char* fn_name) {
    auto out = nb::cast<NDArray2dD>(out_obj);
    if (out.shape(0) != rows || out.shape(1) != cols) {
        throw std::runtime_error(std::string(fn_name) + ": out has wrong shape.");
    }
    return out;
}

inline nb::object extract_eval_objectives(nb::handle eval_out, nb::handle np) {
    nb::object f_obj;
    if (nb::hasattr(eval_out, "F")) {
        f_obj = nb::borrow<nb::object>(eval_out.attr("F"));
    } else if (nb::isinstance<nb::dict>(eval_out)) {
        nb::dict d = nb::cast<nb::dict>(eval_out);
        if (!d.contains("F")) {
            throw std::runtime_error("eval_fn returned dict without key 'F'.");
        }
        f_obj = nb::borrow<nb::object>(d["F"]);
    } else {
        f_obj = nb::borrow<nb::object>(eval_out);
    }
    return np.attr("ascontiguousarray")(f_obj, "float64");
}

inline bool dominates(const double* F, size_t n_obj, size_t i, size_t j) {
    bool any_strict = false;
    for (size_t m = 0; m < n_obj; ++m) {
        const double fi = F[i * n_obj + m];
        const double fj = F[j * n_obj + m];
        if (fi > fj) {
            return false;
        }
        if (fi < fj) {
            any_strict = true;
        }
    }
    return any_strict;
}

std::pair<Fronts, std::vector<int64_t>> fast_non_dominated_sort_impl(const double* F, size_t n, size_t n_obj) {
    Fronts fronts;
    std::vector<int64_t> rank(n, 0);
    if (n == 0) {
        return {fronts, rank};
    }

    std::vector<uint8_t> dom_matrix(n * n, 0);
    std::vector<int64_t> dominated_count(n, 0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            if (dominates(F, n_obj, i, j)) {
                dom_matrix[i * n + j] = 1;
                dominated_count[j] += 1;
            }
        }
    }

    std::vector<size_t> current;
    current.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (dominated_count[i] == 0) {
            current.push_back(i);
        }
    }

    int64_t level = 0;
    while (!current.empty()) {
        fronts.push_back(current);
        for (size_t idx : current) {
            rank[idx] = level;
        }

        for (size_t p : current) {
            for (size_t q = 0; q < n; ++q) {
                if (dom_matrix[p * n + q] != 0) {
                    dom_matrix[p * n + q] = 0;
                    dominated_count[q] -= 1;
                }
            }
            dominated_count[p] = -1;
        }

        std::vector<size_t> next;
        next.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (dominated_count[i] == 0) {
                next.push_back(i);
            }
        }
        current.swap(next);
        level += 1;
    }
    return {fronts, rank};
}

std::vector<double> crowding_distance_impl(const double* F, size_t n, size_t n_obj, const Fronts& fronts) {
    std::vector<double> crowding(n, 0.0);
    for (const auto& front : fronts) {
        if (front.empty()) {
            continue;
        }
        if (front.size() == 1) {
            crowding[front[0]] = std::numeric_limits<double>::infinity();
            continue;
        }

        std::vector<double> d(front.size(), 0.0);
        for (size_t m = 0; m < n_obj; ++m) {
            std::vector<size_t> order(front.size());
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return F[front[a] * n_obj + m] < F[front[b] * n_obj + m];
            });

            const double first_v = F[front[order.front()] * n_obj + m];
            const double last_v = F[front[order.back()] * n_obj + m];
            d[order.front()] = std::numeric_limits<double>::infinity();
            d[order.back()] = std::numeric_limits<double>::infinity();

            const double span = last_v - first_v;
            if (span <= 0.0) {
                continue;
            }
            for (size_t k = 1; k + 1 < order.size(); ++k) {
                const size_t l = order[k - 1];
                const size_t c = order[k];
                const size_t r = order[k + 1];
                const double lv = F[front[l] * n_obj + m];
                const double rv = F[front[r] * n_obj + m];
                d[c] += (rv - lv) / span;
            }
        }

        for (size_t i = 0; i < front.size(); ++i) {
            crowding[front[i]] = d[i];
        }
    }
    return crowding;
}

std::vector<int64_t> select_nsga2_impl(const Fronts& fronts, const std::vector<double>& crowding, size_t pop_size) {
    std::vector<int64_t> selected;
    selected.reserve(pop_size);
    for (const auto& front : fronts) {
        if (front.empty()) {
            continue;
        }
        if (selected.size() + front.size() <= pop_size) {
            for (size_t idx : front) {
                selected.push_back(static_cast<int64_t>(idx));
            }
        } else {
            const size_t rem = pop_size - selected.size();
            std::vector<size_t> order(front.size());
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return crowding[front[a]] > crowding[front[b]];
            });
            for (size_t k = 0; k < rem; ++k) {
                selected.push_back(static_cast<int64_t>(front[order[k]]));
            }
            break;
        }
    }
    return selected;
}

std::vector<size_t> sample_without_replacement(size_t n, size_t k, std::mt19937_64& rng) {
    std::vector<size_t> pool(n);
    std::iota(pool.begin(), pool.end(), 0);
    for (size_t i = 0; i < k; ++i) {
        std::uniform_int_distribution<size_t> d(i, n - 1);
        const size_t j = d(rng);
        std::swap(pool[i], pool[j]);
    }
    return std::vector<size_t>(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(k));
}

std::vector<int64_t> tournament_selection_impl(
    const int64_t* ranks,
    const double* crowd,
    size_t n,
    int pressure,
    uint64_t seed,
    size_t n_parents
) {
    if (pressure <= 0) {
        throw std::runtime_error("pressure must be positive.");
    }
    if (n == 0 || n_parents == 0) {
        return {};
    }
    if (static_cast<size_t>(pressure) > n) {
        throw std::runtime_error("pressure cannot exceed population size.");
    }

    std::mt19937_64 rng(seed);
    std::vector<int64_t> winners(n_parents, 0);
    if (pressure == 1) {
        std::uniform_int_distribution<size_t> di(0, n - 1);
        for (size_t i = 0; i < n_parents; ++i) {
            winners[i] = static_cast<int64_t>(di(rng));
        }
        return winners;
    }

    for (size_t i = 0; i < n_parents; ++i) {
        const auto cand = sample_without_replacement(n, static_cast<size_t>(pressure), rng);
        int64_t min_rank = std::numeric_limits<int64_t>::max();
        for (size_t c : cand) {
            min_rank = std::min(min_rank, ranks[c]);
        }

        std::vector<size_t> best;
        for (size_t c : cand) {
            if (ranks[c] == min_rank) {
                best.push_back(c);
            }
        }
        if (best.size() == 1) {
            winners[i] = static_cast<int64_t>(best[0]);
            continue;
        }

        double max_c = -std::numeric_limits<double>::infinity();
        for (size_t c : best) {
            double v = crowd[c];
            if (std::isnan(v)) {
                v = -std::numeric_limits<double>::infinity();
            }
            if (v > max_c) {
                max_c = v;
            }
        }
        std::vector<size_t> tied;
        for (size_t c : best) {
            double v = crowd[c];
            if (std::isnan(v)) {
                v = -std::numeric_limits<double>::infinity();
            }
            if (v == max_c) {
                tied.push_back(c);
            }
        }
        std::uniform_int_distribution<size_t> di(0, tied.size() - 1);
        winners[i] = static_cast<int64_t>(tied[di(rng)]);
    }
    return winners;
}

double hypervolume_recursive_impl(const std::vector<std::vector<double>>& pts, const std::vector<double>& ref, size_t n_obj);

double hypervolume_2d_impl(const std::vector<std::vector<double>>& pts, const std::vector<double>& ref) {
    if (pts.empty()) {
        return 0.0;
    }
    std::vector<size_t> order(pts.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return pts[a][0] < pts[b][0];
    });

    double hv = 0.0;
    double prev_min = ref[1];
    for (size_t idx : order) {
        const double w = std::max(ref[0] - pts[idx][0], 0.0);
        const double h = std::max(prev_min - pts[idx][1], 0.0);
        hv += w * h;
        prev_min = std::min(prev_min, pts[idx][1]);
    }
    return hv;
}

double hypervolume_3d_impl(const std::vector<std::vector<double>>& pts, const std::vector<double>& ref) {
    std::vector<size_t> order(pts.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return pts[a][2] < pts[b][2];
    });
    double hv = 0.0;
    double prev = ref[2];
    for (size_t e = order.size(); e > 0; --e) {
        const size_t idx = order[e - 1];
        const double h = std::max(prev - pts[idx][2], 0.0);
        if (h <= 0.0) {
            continue;
        }
        std::vector<std::vector<double>> slice;
        slice.reserve(e);
        for (size_t k = 0; k < e; ++k) {
            const auto& p = pts[order[k]];
            slice.push_back({p[0], p[1]});
        }
        hv += hypervolume_2d_impl(slice, {ref[0], ref[1]}) * h;
        prev = pts[idx][2];
    }
    return hv;
}

double hypervolume_recursive_impl(const std::vector<std::vector<double>>& pts, const std::vector<double>& ref, size_t n_obj) {
    if (pts.empty()) {
        return 0.0;
    }
    if (n_obj == 1) {
        double best = 0.0;
        for (const auto& p : pts) {
            best = std::max(best, std::max(ref[0] - p[0], 0.0));
        }
        return best;
    }
    std::vector<size_t> order(pts.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return pts[a][n_obj - 1] < pts[b][n_obj - 1];
    });
    double hv = 0.0;
    double bound = ref[n_obj - 1];
    for (size_t e = order.size(); e > 0; --e) {
        const size_t idx = order[e - 1];
        const double cur = pts[idx][n_obj - 1];
        const double h = bound - cur;
        if (h > 0.0) {
            std::vector<std::vector<double>> reduced;
            reduced.reserve(e);
            for (size_t k = 0; k < e; ++k) {
                std::vector<double> p;
                p.reserve(n_obj - 1);
                for (size_t m = 0; m + 1 < n_obj; ++m) {
                    p.push_back(pts[order[k]][m]);
                }
                reduced.push_back(std::move(p));
            }
            std::vector<double> r(ref.begin(), ref.begin() + static_cast<std::ptrdiff_t>(n_obj - 1));
            hv += hypervolume_recursive_impl(reduced, r, n_obj - 1) * h;
            bound = cur;
        }
    }
    return hv;
}

double hypervolume_impl(const double* points, size_t n, size_t n_obj, const double* ref) {
    if (n == 0) {
        return 0.0;
    }
    std::vector<std::vector<double>> pts(n, std::vector<double>(n_obj, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t m = 0; m < n_obj; ++m) {
            pts[i][m] = points[i * n_obj + m];
        }
    }
    std::vector<double> ref_v(ref, ref + n_obj);
    if (n_obj == 1) {
        return hypervolume_recursive_impl(pts, ref_v, 1);
    }
    if (n_obj == 2) {
        return hypervolume_2d_impl(pts, ref_v);
    }
    if (n_obj == 3) {
        return hypervolume_3d_impl(pts, ref_v);
    }
    return hypervolume_recursive_impl(pts, ref_v, n_obj);
}

std::vector<double> hypervolume_contributions_2d_impl(const double* points, size_t n, const double* ref) {
    std::vector<double> out(n, 0.0);
    if (n == 0) {
        return out;
    }

    struct Entry {
        double x;
        double y;
        size_t orig;
    };

    std::vector<Entry> sorted(n);
    for (size_t i = 0; i < n; ++i) {
        sorted[i] = Entry{points[i * 2], points[i * 2 + 1], i};
    }
    std::stable_sort(sorted.begin(), sorted.end(), [](const Entry& a, const Entry& b) {
        if (a.x < b.x) {
            return true;
        }
        if (a.x > b.x) {
            return false;
        }
        return a.y < b.y;
    });

    std::vector<double> ux;
    std::vector<double> uy;
    std::vector<size_t> counts;
    std::vector<size_t> inverse(n, 0);
    ux.reserve(n);
    uy.reserve(n);
    counts.reserve(n);

    for (const Entry& e : sorted) {
        bool is_new = ux.empty() || e.x != ux.back() || e.y != uy.back();
        if (is_new) {
            ux.push_back(e.x);
            uy.push_back(e.y);
            counts.push_back(0);
        }
        const size_t gid = ux.size() - 1;
        counts[gid] += 1;
        inverse[e.orig] = gid;
    }

    const size_t u = ux.size();
    std::vector<double> unique_contrib(u, 0.0);
    std::vector<size_t> nd_ids;
    nd_ids.reserve(u);

    double prev_min = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < u; ++i) {
        if (uy[i] < prev_min) {
            nd_ids.push_back(i);
            prev_min = uy[i];
        }
    }

    for (size_t t = 0; t < nd_ids.size(); ++t) {
        const size_t gid = nd_ids[t];
        const double x = ux[gid];
        const double y = uy[gid];
        const double x_next = (t + 1 < nd_ids.size()) ? ux[nd_ids[t + 1]] : ref[0];
        const double y_prev = (t == 0) ? ref[1] : uy[nd_ids[t - 1]];
        const double w = std::max(x_next - x, 0.0);
        const double h = std::max(y_prev - y, 0.0);
        unique_contrib[gid] = w * h;
    }

    for (size_t i = 0; i < n; ++i) {
        const size_t gid = inverse[i];
        out[i] = (counts[gid] > 1) ? 0.0 : unique_contrib[gid];
    }
    return out;
}

std::vector<double> hypervolume_contributions_impl(const double* points, size_t n, size_t n_obj, const double* ref) {
    std::vector<double> out(n, 0.0);
    if (n == 0) {
        return out;
    }
    if (n_obj == 2) {
        return hypervolume_contributions_2d_impl(points, n, ref);
    }
    const double full = hypervolume_impl(points, n, n_obj, ref);
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> wo;
        wo.reserve((n - 1) * n_obj);
        for (size_t r = 0; r < n; ++r) {
            if (r == i) {
                continue;
            }
            for (size_t m = 0; m < n_obj; ++m) {
                wo.push_back(points[r * n_obj + m]);
            }
        }
        const double hv_wo = hypervolume_impl(wo.data(), n - 1, n_obj, ref);
        out[i] = full - hv_wo;
    }
    return out;
}

std::vector<uint8_t> dominance_matrix_impl(const double* F, size_t n, size_t n_obj) {
    std::vector<uint8_t> dom(n * n, 0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            dom[i * n + j] = dominates(F, n_obj, i, j) ? 1 : 0;
        }
    }
    return dom;
}

std::pair<std::vector<double>, std::vector<double>> spea2_fitness_impl(
    const double* F,
    size_t n,
    size_t n_obj,
    const uint8_t* dom,
    int k
) {
    std::vector<double> fitness(n, 0.0);
    std::vector<double> dist(n * n, 0.0);
    if (n == 0) {
        return {fitness, dist};
    }
    if (n > 1) {
        k = std::min(k, static_cast<int>(n - 1));
    } else {
        k = 1;
    }

    std::vector<double> strength(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < n; ++j) {
            s += dom[i * n + j] ? 1.0 : 0.0;
        }
        strength[i] = s;
    }

    std::vector<double> raw(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double rf = 0.0;
        for (size_t j = 0; j < n; ++j) {
            if (dom[j * n + i]) {
                rf += strength[j];
            }
        }
        raw[i] = rf;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double s = 0.0;
            for (size_t m = 0; m < n_obj; ++m) {
                const double d = F[i * n_obj + m] - F[j * n_obj + m];
                s += d * d;
            }
            const double d = std::sqrt(s);
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }

    std::vector<double> density(n, 0.0);
    if (n == 1) {
        density[0] = 0.0;
    } else {
        for (size_t i = 0; i < n; ++i) {
            std::vector<double> row(n, 0.0);
            for (size_t j = 0; j < n; ++j) {
                row[j] = dist[i * n + j];
            }
            std::stable_sort(row.begin(), row.end());
            const double sigma_k = row[static_cast<size_t>(k)];
            density[i] = 1.0 / (sigma_k + 2.0);
        }
    }

    for (size_t i = 0; i < n; ++i) {
        fitness[i] = raw[i] + density[i];
    }
    return {fitness, dist};
}

std::vector<double> spea2_raw_fitness_from_dom_impl(const uint8_t* dom, size_t n) {
    std::vector<double> raw(n, 0.0);
    if (n == 0) {
        return raw;
    }
    std::vector<double> strength(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < n; ++j) {
            s += dom[i * n + j] ? 1.0 : 0.0;
        }
        strength[i] = s;
    }
    for (size_t i = 0; i < n; ++i) {
        double rf = 0.0;
        for (size_t j = 0; j < n; ++j) {
            if (dom[j * n + i]) {
                rf += strength[j];
            }
        }
        raw[i] = rf;
    }
    return raw;
}

std::vector<size_t> truncate_by_distance_indices_impl(const std::vector<double>& dist, size_t n, size_t keep, int k) {
    std::vector<size_t> candidates(n, 0);
    std::iota(candidates.begin(), candidates.end(), 0);
    if (n <= keep) {
        return candidates;
    }
    if (k < 1) {
        k = 1;
    }
    if (n > 1) {
        k = std::min(k, static_cast<int>(n - 1));
    }

    while (candidates.size() > keep) {
        size_t remove_pos = 0;
        double best = std::numeric_limits<double>::infinity();
        for (size_t pos = 0; pos < candidates.size(); ++pos) {
            const size_t i = candidates[pos];
            std::vector<double> row;
            row.reserve(candidates.size() - 1);
            for (size_t q = 0; q < candidates.size(); ++q) {
                if (q == pos) {
                    continue;
                }
                const size_t j = candidates[q];
                row.push_back(dist[i * n + j]);
            }
            if (row.empty()) {
                continue;
            }
            const size_t kk = std::min(static_cast<size_t>(k), row.size() - 1);
            std::nth_element(row.begin(), row.begin() + static_cast<std::ptrdiff_t>(kk), row.end());
            const double kth = row[kk];
            if (kth < best) {
                best = kth;
                remove_pos = pos;
            }
        }
        candidates.erase(candidates.begin() + static_cast<std::ptrdiff_t>(remove_pos));
    }
    return candidates;
}

std::vector<int64_t> spea2_environmental_selection_indices_impl(
    const double* F,
    size_t n,
    size_t n_obj,
    size_t archive_size,
    int k
) {
    std::vector<int64_t> selected_i64;
    if (n == 0) {
        return selected_i64;
    }
    if (archive_size >= n) {
        selected_i64.resize(n, 0);
        for (size_t i = 0; i < n; ++i) {
            selected_i64[i] = static_cast<int64_t>(i);
        }
        return selected_i64;
    }

    const auto dom = dominance_matrix_impl(F, n, n_obj);
    const auto raw = spea2_raw_fitness_from_dom_impl(dom.data(), n);

    std::vector<double> unique = raw;
    std::stable_sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());

    std::vector<size_t> selected;
    selected.reserve(archive_size);

    for (double fit : unique) {
        std::vector<size_t> front;
        front.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (raw[i] == fit) {
                front.push_back(i);
            }
        }
        if (selected.size() + front.size() <= archive_size) {
            selected.insert(selected.end(), front.begin(), front.end());
            continue;
        }

        const size_t remaining = archive_size - selected.size();
        if (remaining == 0 || front.empty()) {
            break;
        }

        const size_t m = front.size();
        std::vector<double> dist(m * m, 0.0);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = i + 1; j < m; ++j) {
                double s = 0.0;
                for (size_t obj = 0; obj < n_obj; ++obj) {
                    const double d = F[front[i] * n_obj + obj] - F[front[j] * n_obj + obj];
                    s += d * d;
                }
                const double dij = std::sqrt(s);
                dist[i * m + j] = dij;
                dist[j * m + i] = dij;
            }
        }

        const auto keep_local = truncate_by_distance_indices_impl(dist, m, remaining, k);
        for (size_t lid : keep_local) {
            selected.push_back(front[lid]);
        }
        break;
    }

    selected_i64.resize(selected.size(), 0);
    for (size_t i = 0; i < selected.size(); ++i) {
        selected_i64[i] = static_cast<int64_t>(selected[i]);
    }
    return selected_i64;
}

std::vector<double> epsilon_indicator_impl(const double* F, size_t n, size_t n_obj) {
    std::vector<double> out(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double mx = -std::numeric_limits<double>::infinity();
            for (size_t m = 0; m < n_obj; ++m) {
                mx = std::max(mx, F[j * n_obj + m] - F[i * n_obj + m]);
            }
            out[i * n + j] = mx;
        }
    }
    return out;
}

std::vector<double> hv_indicator_impl(const double* F, size_t n, size_t n_obj, const double* ref) {
    std::vector<double> out(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            std::vector<double> pair;
            pair.reserve(2 * n_obj);
            for (size_t m = 0; m < n_obj; ++m) {
                pair.push_back(F[i * n_obj + m]);
            }
            for (size_t m = 0; m < n_obj; ++m) {
                pair.push_back(F[j * n_obj + m]);
            }
            const double hv_pair = hypervolume_impl(pair.data(), 2, n_obj, ref);
            const double hv_j = hypervolume_impl(F + j * n_obj, 1, n_obj, ref);
            out[i * n + j] = hv_j - hv_pair;
        }
    }
    return out;
}

double safe_exp_indicator_term(double value, double kappa) {
    const double denom = (std::abs(kappa) > 1.0e-12) ? kappa : 1.0e-12;
    const double term = std::exp(-value / denom);
    if (!std::isfinite(term)) {
        return 0.0;
    }
    return term;
}

std::vector<double> ibea_fitness_from_indicator_impl(const std::vector<double>& ind, size_t n, double kappa) {
    std::vector<double> fitness(n, 0.0);
    if (n == 0) {
        return fitness;
    }
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            sum += safe_exp_indicator_term(ind[i * n + j], kappa);
        }
        fitness[i] = -sum;
    }
    return fitness;
}

std::pair<std::vector<int64_t>, std::vector<double>> ibea_environmental_selection_indices_impl(
    const double* F,
    size_t n,
    size_t n_obj,
    size_t pop_size,
    const double* ref,
    const std::string& kind,
    double kappa
) {
    std::vector<int64_t> selected;
    selected.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        selected.push_back(static_cast<int64_t>(i));
    }
    if (n == 0 || pop_size == 0) {
        return {std::vector<int64_t>(), std::vector<double>()};
    }

    std::vector<double> indicator;
    if (kind == "hypervolume") {
        indicator = hv_indicator_impl(F, n, n_obj, ref);
    } else {
        indicator = epsilon_indicator_impl(F, n, n_obj);
    }

    std::vector<double> fitness = ibea_fitness_from_indicator_impl(indicator, n, kappa);
    if (pop_size >= n) {
        return {selected, fitness};
    }

    while (selected.size() > pop_size) {
        size_t worst_pos = 0;
        for (size_t pos = 1; pos < fitness.size(); ++pos) {
            if (fitness[pos] < fitness[worst_pos]) {
                worst_pos = pos;
            }
        }
        const int64_t worst_idx = selected[worst_pos];
        for (size_t pos = 0; pos < selected.size(); ++pos) {
            if (pos == worst_pos) {
                continue;
            }
            const int64_t idx = selected[pos];
            fitness[pos] += safe_exp_indicator_term(
                indicator[static_cast<size_t>(idx) * n + static_cast<size_t>(worst_idx)],
                kappa
            );
        }
        selected.erase(selected.begin() + static_cast<std::ptrdiff_t>(worst_pos));
        fitness.erase(fitness.begin() + static_cast<std::ptrdiff_t>(worst_pos));
    }
    return {selected, fitness};
}

std::vector<double> sbx_crossover_impl(
    const double* parents,
    size_t n_parents,
    size_t n_var,
    double prob,
    double eta,
    const std::vector<double>& lower,
    const std::vector<double>& upper,
    uint64_t seed,
    double prob_var
);

std::vector<double> polynomial_mutation_impl(
    const double* X,
    size_t n_ind,
    size_t n_var,
    double prob,
    double eta,
    const std::vector<double>& lower,
    const std::vector<double>& upper,
    uint64_t seed
);

std::vector<double> smsemoa_generate_offspring_impl(
    const double* X,
    const double* F,
    size_t n,
    size_t n_var,
    size_t n_obj,
    const std::string& selection,
    int pressure,
    const std::vector<double>& lower,
    const std::vector<double>& upper,
    double sbx_prob,
    double sbx_eta,
    double pm_prob,
    double pm_eta,
    uint64_t seed
) {
    if (n == 0) {
        throw std::runtime_error("X cannot be empty.");
    }

    std::mt19937_64 rng(seed);
    std::vector<int64_t> parent_idx(2, 0);
    if (selection == "tournament" && n > 1) {
        auto [fronts, ranks] = fast_non_dominated_sort_impl(F, n, n_obj);
        const auto crowd = crowding_distance_impl(F, n, n_obj, fronts);
        parent_idx = tournament_selection_impl(ranks.data(), crowd.data(), n, pressure, static_cast<uint64_t>(rng()), 2);
    } else if (selection == "tournament") {
        parent_idx[0] = 0;
        parent_idx[1] = 0;
    } else {
        std::uniform_int_distribution<size_t> di(0, n - 1);
        parent_idx[0] = static_cast<int64_t>(di(rng));
        parent_idx[1] = static_cast<int64_t>(di(rng));
    }

    std::vector<double> parents(2 * n_var, 0.0);
    for (size_t i = 0; i < 2; ++i) {
        const size_t src = static_cast<size_t>(parent_idx[i]);
        std::copy(
            X + static_cast<std::ptrdiff_t>(src * n_var),
            X + static_cast<std::ptrdiff_t>((src + 1) * n_var),
            parents.begin() + static_cast<std::ptrdiff_t>(i * n_var)
        );
    }

    auto children = sbx_crossover_impl(parents.data(), 2, n_var, sbx_prob, sbx_eta, lower, upper, static_cast<uint64_t>(rng()), 0.5);
    children.resize(2 * n_var);
    return polynomial_mutation_impl(children.data(), 1, n_var, pm_prob, pm_eta, lower, upper, static_cast<uint64_t>(rng()));
}

std::vector<double> spea2_generate_offspring_impl(
    const double* X,
    const double* F,
    size_t n,
    size_t n_var,
    size_t n_obj,
    size_t n_offspring,
    int k_neighbors,
    const std::vector<double>& lower,
    const std::vector<double>& upper,
    double sbx_prob,
    double sbx_eta,
    double pm_prob,
    double pm_eta,
    uint64_t seed
) {
    if (n_offspring == 0) {
        return {};
    }
    if (n == 0) {
        throw std::runtime_error("F cannot be empty.");
    }

    std::mt19937_64 rng(seed);
    const auto dom = dominance_matrix_impl(F, n, n_obj);
    const auto raw = spea2_raw_fitness_from_dom_impl(dom.data(), n);

    int k = std::max(1, k_neighbors);
    if (n > 1) {
        k = std::min(k, static_cast<int>(n - 1));
    } else {
        k = 1;
    }

    std::vector<double> density(n, 0.0);
    if (n == 1) {
        density[0] = std::numeric_limits<double>::infinity();
    } else {
        for (size_t i = 0; i < n; ++i) {
            std::vector<double> row(n, 0.0);
            for (size_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (size_t m = 0; m < n_obj; ++m) {
                    const double d = F[i * n_obj + m] - F[j * n_obj + m];
                    s += d * d;
                }
                row[j] = std::sqrt(s);
            }
            std::stable_sort(row.begin(), row.end());
            density[i] = row[static_cast<size_t>(k)];
        }
    }

    std::vector<size_t> order(n, 0);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return raw[a] < raw[b];
    });
    std::vector<int64_t> ranks(n, 0);
    for (size_t pos = 0; pos < n; ++pos) {
        ranks[order[pos]] = static_cast<int64_t>(pos);
    }

    const size_t parent_count = n_offspring * 2;
    std::vector<int64_t> parent_idx(parent_count, 0);
    if (n > 1) {
        parent_idx = tournament_selection_impl(ranks.data(), density.data(), n, 2, static_cast<uint64_t>(rng()), parent_count);
    }

    std::vector<double> parents(parent_count * n_var, 0.0);
    for (size_t i = 0; i < parent_count; ++i) {
        const size_t src = static_cast<size_t>(parent_idx[i]);
        std::copy(
            X + static_cast<std::ptrdiff_t>(src * n_var),
            X + static_cast<std::ptrdiff_t>((src + 1) * n_var),
            parents.begin() + static_cast<std::ptrdiff_t>(i * n_var)
        );
    }

    const auto crossed = sbx_crossover_impl(
        parents.data(),
        parent_count,
        n_var,
        sbx_prob,
        sbx_eta,
        lower,
        upper,
        static_cast<uint64_t>(rng()),
        0.5
    );
    std::vector<double> offspring(n_offspring * n_var, 0.0);
    for (size_t i = 0; i < n_offspring; ++i) {
        const size_t src_row = i * 2;
        std::copy(
            crossed.begin() + static_cast<std::ptrdiff_t>(src_row * n_var),
            crossed.begin() + static_cast<std::ptrdiff_t>((src_row + 1) * n_var),
            offspring.begin() + static_cast<std::ptrdiff_t>(i * n_var)
        );
    }

    return polynomial_mutation_impl(
        offspring.data(),
        n_offspring,
        n_var,
        pm_prob,
        pm_eta,
        lower,
        upper,
        static_cast<uint64_t>(rng())
    );
}

std::pair<std::vector<double>, std::vector<double>> normalize_bounds_impl(
    const double* xl,
    size_t xl_n,
    const double* xu,
    size_t xu_n,
    size_t n_var
) {
    std::vector<double> lower(n_var, 0.0);
    std::vector<double> upper(n_var, 0.0);
    if (xl_n == 1) {
        std::fill(lower.begin(), lower.end(), xl[0]);
    } else if (xl_n == n_var) {
        std::copy(xl, xl + xl_n, lower.begin());
    } else {
        throw std::runtime_error("xl must be size 1 or n_var.");
    }
    if (xu_n == 1) {
        std::fill(upper.begin(), upper.end(), xu[0]);
    } else if (xu_n == n_var) {
        std::copy(xu, xu + xu_n, upper.begin());
    } else {
        throw std::runtime_error("xu must be size 1 or n_var.");
    }
    return {lower, upper};
}

std::vector<double> sbx_crossover_impl(
    const double* parents,
    size_t n_parents,
    size_t n_var,
    double prob,
    double eta,
    const std::vector<double>& lower,
    const std::vector<double>& upper,
    uint64_t seed,
    double prob_var
) {
    if (n_parents == 0) {
        return {};
    }

    std::vector<double> work;
    size_t effective_n = n_parents;
    if (n_parents % 2 != 0) {
        effective_n = n_parents + 1;
        work.resize(effective_n * n_var);
        std::copy(parents, parents + n_parents * n_var, work.begin());
        const double* last = parents + (n_parents - 1) * n_var;
        std::copy(last, last + n_var, work.begin() + (effective_n - 1) * n_var);
    } else {
        work.assign(parents, parents + n_parents * n_var);
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> ur(0.0, 1.0);
    constexpr double eps = 1.0e-14;

    for (size_t i = 0; i < effective_n; i += 2) {
        if (ur(rng) > prob) {
            continue;
        }
        for (size_t j = 0; j < n_var; ++j) {
            if (ur(rng) > prob_var) {
                continue;
            }
            double y1 = work[i * n_var + j];
            double y2 = work[(i + 1) * n_var + j];
            const double yl = lower[j];
            const double yu = upper[j];
            if (std::abs(y1 - y2) < eps || yl >= yu) {
                continue;
            }
            double y1v = y1;
            double y2v = y2;
            if (y1v > y2v) {
                std::swap(y1v, y2v);
            }

            double beta = 1.0 + (2.0 * (y1v - yl) / (y2v - y1v));
            double alpha = 2.0 - std::pow(beta, -(eta + 1.0));
            const double rnd = ur(rng);
            double betaq;
            if (rnd <= (1.0 / alpha)) {
                betaq = std::pow(rnd * alpha, 1.0 / (eta + 1.0));
            } else {
                betaq = std::pow(1.0 / (2.0 - rnd * alpha), 1.0 / (eta + 1.0));
            }
            double c1 = 0.5 * ((y1v + y2v) - betaq * (y2v - y1v));

            beta = 1.0 + (2.0 * (yu - y2v) / (y2v - y1v));
            alpha = 2.0 - std::pow(beta, -(eta + 1.0));
            if (rnd <= (1.0 / alpha)) {
                betaq = std::pow(rnd * alpha, 1.0 / (eta + 1.0));
            } else {
                betaq = std::pow(1.0 / (2.0 - rnd * alpha), 1.0 / (eta + 1.0));
            }
            double c2 = 0.5 * ((y1v + y2v) + betaq * (y2v - y1v));

            c1 = std::min(std::max(c1, yl), yu);
            c2 = std::min(std::max(c2, yl), yu);
            if (ur(rng) <= 0.5) {
                work[i * n_var + j] = c2;
                work[(i + 1) * n_var + j] = c1;
            } else {
                work[i * n_var + j] = c1;
                work[(i + 1) * n_var + j] = c2;
            }
        }
    }
    return work;
}

std::vector<double> polynomial_mutation_impl(
    const double* X,
    size_t n_ind,
    size_t n_var,
    double prob,
    double eta,
    const std::vector<double>& lower,
    const std::vector<double>& upper,
    uint64_t seed
) {
    std::vector<double> out(X, X + n_ind * n_var);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> ur(0.0, 1.0);
    const double mut_pow = 1.0 / (eta + 1.0);
    for (size_t i = 0; i < n_ind; ++i) {
        for (size_t j = 0; j < n_var; ++j) {
            if (ur(rng) > prob) {
                continue;
            }
            double y = out[i * n_var + j];
            const double yl = lower[j];
            const double yu = upper[j];
            if (yl >= yu) {
                continue;
            }
            const double delta1 = (y - yl) / (yu - yl);
            const double delta2 = (yu - y) / (yu - yl);
            const double rnd = ur(rng);
            double deltaq;
            if (rnd <= 0.5) {
                const double xy = 1.0 - delta1;
                const double val = 2.0 * rnd + (1.0 - 2.0 * rnd) * std::pow(xy, eta + 1.0);
                deltaq = std::pow(val, mut_pow) - 1.0;
            } else {
                const double xy = 1.0 - delta2;
                const double val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * std::pow(xy, eta + 1.0);
                deltaq = 1.0 - std::pow(val, mut_pow);
            }
            y += deltaq * (yu - yl);
            out[i * n_var + j] = std::min(std::max(y, yl), yu);
        }
    }
    return out;
}

std::vector<double> generate_offspring_impl(
    const double* X,
    const double* F,
    size_t n,
    size_t n_var,
    size_t n_obj,
    size_t n_offspring,
    const std::vector<double>& lower,
    const std::vector<double>& upper,
    int pressure,
    double sbx_prob,
    double sbx_eta,
    double pm_prob,
    double pm_eta,
    uint64_t seed
) {
    if (n_offspring == 0) {
        return {};
    }

    auto [fronts, ranks] = fast_non_dominated_sort_impl(F, n, n_obj);
    const auto crowd = crowding_distance_impl(F, n, n_obj, fronts);
    std::mt19937_64 rng(seed);

    size_t parent_count = n_offspring;
    if (parent_count % 2 != 0) {
        parent_count += 1;
    }
    auto parents_idx = tournament_selection_impl(ranks.data(), crowd.data(), n, pressure, seed, parent_count);

    std::vector<double> parents(parent_count * n_var, 0.0);
    for (size_t i = 0; i < parent_count; ++i) {
        const size_t p = static_cast<size_t>(parents_idx[i]);
        std::copy(
            X + static_cast<std::ptrdiff_t>(p * n_var),
            X + static_cast<std::ptrdiff_t>((p + 1) * n_var),
            parents.begin() + static_cast<std::ptrdiff_t>(i * n_var)
        );
    }

    auto children = sbx_crossover_impl(
        parents.data(),
        parent_count,
        n_var,
        sbx_prob,
        sbx_eta,
        lower,
        upper,
        static_cast<uint64_t>(rng()),
        0.5
    );
    children.resize(n_offspring * n_var);
    return polynomial_mutation_impl(
        children.data(),
        n_offspring,
        n_var,
        pm_prob,
        pm_eta,
        lower,
        upper,
        static_cast<uint64_t>(rng())
    );
}

std::pair<std::vector<double>, std::vector<double>> nsga2_survival_flat_impl(
    const std::vector<double>& X,
    const std::vector<double>& F,
    const std::vector<double>& X_off,
    const std::vector<double>& F_off,
    size_t n_var,
    size_t n_obj,
    size_t pop_size,
    std::vector<int64_t>* selected_out
) {
    if (n_var == 0 || n_obj == 0) {
        throw std::runtime_error("n_var and n_obj must be positive in nsga2_survival_flat_impl.");
    }
    if (X.size() % n_var != 0 || X_off.size() % n_var != 0 || F.size() % n_obj != 0 || F_off.size() % n_obj != 0) {
        throw std::runtime_error("flat buffer size mismatch in nsga2_survival_flat_impl.");
    }

    const size_t n = X.size() / n_var;
    const size_t n_off = X_off.size() / n_var;
    if ((F.size() / n_obj) != n || (F_off.size() / n_obj) != n_off) {
        throw std::runtime_error("row mismatch between decision/objective buffers in nsga2_survival_flat_impl.");
    }

    const size_t n_comb = n + n_off;
    if (pop_size > n_comb) {
        throw std::runtime_error("pop_size cannot exceed combined population in nsga2_survival_flat_impl.");
    }

    std::vector<double> Xc(n_comb * n_var, 0.0);
    std::vector<double> Fc(n_comb * n_obj, 0.0);
    std::copy(X.begin(), X.end(), Xc.begin());
    std::copy(X_off.begin(), X_off.end(), Xc.begin() + static_cast<std::ptrdiff_t>(n * n_var));
    std::copy(F.begin(), F.end(), Fc.begin());
    std::copy(F_off.begin(), F_off.end(), Fc.begin() + static_cast<std::ptrdiff_t>(n * n_obj));

    const auto [fronts, _r] = fast_non_dominated_sort_impl(Fc.data(), n_comb, n_obj);
    const auto crowd = crowding_distance_impl(Fc.data(), n_comb, n_obj, fronts);
    const auto sel = select_nsga2_impl(fronts, crowd, pop_size);

    std::vector<double> Xnew(pop_size * n_var, 0.0);
    std::vector<double> Fnew(pop_size * n_obj, 0.0);
    for (size_t i = 0; i < pop_size; ++i) {
        const size_t src = static_cast<size_t>(sel[i]);
        std::copy(
            Xc.begin() + static_cast<std::ptrdiff_t>(src * n_var),
            Xc.begin() + static_cast<std::ptrdiff_t>((src + 1) * n_var),
            Xnew.begin() + static_cast<std::ptrdiff_t>(i * n_var)
        );
        std::copy(
            Fc.begin() + static_cast<std::ptrdiff_t>(src * n_obj),
            Fc.begin() + static_cast<std::ptrdiff_t>((src + 1) * n_obj),
            Fnew.begin() + static_cast<std::ptrdiff_t>(i * n_obj)
        );
    }

    if (selected_out != nullptr) {
        *selected_out = sel;
    }
    return {std::move(Xnew), std::move(Fnew)};
}

} // namespace

NB_MODULE(_core, m) {
    m.doc() = "VAMOS++ native extension module";

    m.def("is_native_backend", []() { return true; });
    m.def("backend_info", []() {
        nb::dict info;
        info["backend"] = "nanobind-core";
        info["native"] = true;
        return info;
    });

    m.def("fast_non_dominated_sort", [](NDArray2dDConst F) {
        const auto [fronts, ranks] = fast_non_dominated_sort_impl(F.data(), F.shape(0), F.shape(1));
        return nb::make_tuple(fronts, ranks);
    });

    m.def(
        "crowding_distance",
        [](NDArray2dDConst F, nb::object fronts_obj) {
            Fronts fronts;
            if (fronts_obj.is_none()) {
                fronts = fast_non_dominated_sort_impl(F.data(), F.shape(0), F.shape(1)).first;
            } else {
                fronts = nb::cast<Fronts>(fronts_obj);
            }
            return crowding_distance_impl(F.data(), F.shape(0), F.shape(1), fronts);
        },
        nb::arg("F"),
        nb::arg("fronts") = nb::none()
    );

    m.def("nsga2_ranking", [](NDArray2dDConst F) {
        const auto [fronts, ranks] = fast_non_dominated_sort_impl(F.data(), F.shape(0), F.shape(1));
        const auto crowd = crowding_distance_impl(F.data(), F.shape(0), F.shape(1), fronts);
        return nb::make_tuple(ranks, crowd);
    });

    m.def("tournament_selection", [](NDArray1dIConst ranks, NDArray1dDConst crowding, int pressure, uint64_t seed, int n_parents) {
        const size_t n = ranks.shape(0);
        if (n != crowding.shape(0)) {
            throw std::runtime_error("ranks and crowding must have same shape.");
        }
        return tournament_selection_impl(ranks.data(), crowding.data(), n, pressure, seed, static_cast<size_t>(n_parents));
    });

    m.def(
        "sbx_crossover",
        [](NDArray2dDConst X_parents, double prob, double eta, NDArray1dDConst xl, NDArray1dDConst xu, uint64_t seed, double prob_var) {
            const size_t n_parents = X_parents.shape(0);
            const size_t n_var = X_parents.shape(1);
            const auto [lower, upper] = normalize_bounds_impl(xl.data(), xl.shape(0), xu.data(), xu.shape(0), n_var);
            const auto out = sbx_crossover_impl(X_parents.data(), n_parents, n_var, prob, eta, lower, upper, seed, prob_var);

            const size_t n_out = (n_parents % 2 == 0) ? n_parents : (n_parents + 1);
            std::vector<std::vector<double>> arr(n_out, std::vector<double>(n_var, 0.0));
            for (size_t i = 0; i < n_out; ++i) {
                for (size_t j = 0; j < n_var; ++j) {
                    arr[i][j] = out[i * n_var + j];
                }
            }
            return arr;
        },
        nb::arg("X_parents"),
        nb::arg("prob"),
        nb::arg("eta"),
        nb::arg("xl"),
        nb::arg("xu"),
        nb::arg("seed"),
        nb::arg("prob_var") = 0.5
    );

    m.def(
        "polynomial_mutation",
        [](NDArray2dDConst X, double prob, double eta, NDArray1dDConst xl, NDArray1dDConst xu, uint64_t seed, bool in_place) {
            const size_t n_ind = X.shape(0);
            const size_t n_var = X.shape(1);
            const auto [lower, upper] = normalize_bounds_impl(xl.data(), xl.shape(0), xu.data(), xu.shape(0), n_var);
            const auto out = polynomial_mutation_impl(X.data(), n_ind, n_var, prob, eta, lower, upper, seed);
            (void)in_place;

            std::vector<std::vector<double>> arr(n_ind, std::vector<double>(n_var, 0.0));
            for (size_t i = 0; i < n_ind; ++i) {
                for (size_t j = 0; j < n_var; ++j) {
                    arr[i][j] = out[i * n_var + j];
                }
            }
            return arr;
        },
        nb::arg("X"),
        nb::arg("prob"),
        nb::arg("eta"),
        nb::arg("xl"),
        nb::arg("xu"),
        nb::arg("seed"),
        nb::arg("in_place") = false
    );

    m.def(
        "nsga2_survival",
        [](NDArray2dDConst X, NDArray2dDConst F, NDArray2dDConst X_off, NDArray2dDConst F_off, int pop_size, bool return_indices) -> nb::tuple {
            const size_t n = X.shape(0);
            const size_t n_var = X.shape(1);
            const size_t n_off = X_off.shape(0);
            const size_t n_obj = F.shape(1);
            if (F.shape(0) != n || F_off.shape(0) != n_off || X_off.shape(1) != n_var || F_off.shape(1) != n_obj) {
                throw std::runtime_error("shape mismatch in nsga2_survival.");
            }
            if (pop_size < 0) {
                throw std::runtime_error("pop_size must be non-negative in nsga2_survival.");
            }
            const size_t keep = static_cast<size_t>(pop_size);
            std::vector<double> Xv(X.data(), X.data() + n * n_var);
            std::vector<double> Fv(F.data(), F.data() + n * n_obj);
            std::vector<double> Xoffv(X_off.data(), X_off.data() + n_off * n_var);
            std::vector<double> Foffv(F_off.data(), F_off.data() + n_off * n_obj);

            std::vector<int64_t> sel;
            auto [x_new_flat, f_new_flat] = nsga2_survival_flat_impl(Xv, Fv, Xoffv, Foffv, n_var, n_obj, keep, return_indices ? &sel : nullptr);
            nb::object np = np_module();
            nb::object x_new = make_float64_array_2d(np, x_new_flat, keep, n_var);
            nb::object f_new = make_float64_array_2d(np, f_new_flat, keep, n_obj);

            if (return_indices) {
                return nb::make_tuple(x_new, f_new, make_int64_array_1d(np, sel));
            }
            return nb::make_tuple(x_new, f_new);
        },
        nb::arg("X"),
        nb::arg("F"),
        nb::arg("X_off"),
        nb::arg("F_off"),
        nb::arg("pop_size"),
        nb::arg("return_indices") = false
    );

    m.def("hypervolume", [](NDArray2dDConst points, NDArray1dDConst reference_point) {
        const size_t n = points.shape(0);
        const size_t n_obj = points.shape(1);
        if (reference_point.shape(0) != n_obj) {
            throw std::runtime_error("reference_point dimensionality mismatch.");
        }
        std::vector<double> ref(reference_point.data(), reference_point.data() + n_obj);
        for (size_t i = 0; i < n; ++i) {
            for (size_t m = 0; m < n_obj; ++m) {
                ref[m] = std::max(ref[m], points.data()[i * n_obj + m] + 1.0e-9);
            }
        }
        return hypervolume_impl(points.data(), n, n_obj, ref.data());
    });

    m.def("hypervolume_contributions", [](NDArray2dDConst points, NDArray1dDConst reference_point) {
        const size_t n = points.shape(0);
        const size_t n_obj = points.shape(1);
        if (reference_point.shape(0) != n_obj) {
            throw std::runtime_error("reference_point dimensionality mismatch.");
        }
        std::vector<double> ref(reference_point.data(), reference_point.data() + n_obj);
        for (size_t i = 0; i < n; ++i) {
            for (size_t m = 0; m < n_obj; ++m) {
                ref[m] = std::max(ref[m], points.data()[i * n_obj + m] + 1.0e-9);
            }
        }
        return hypervolume_contributions_impl(points.data(), n, n_obj, ref.data());
    });

    m.def("smsemoa_remove_index", [](NDArray2dDConst F_combined, NDArray1dDConst reference_point) {
        const size_t n = F_combined.shape(0);
        const size_t n_obj = F_combined.shape(1);
        if (reference_point.shape(0) != n_obj) {
            throw std::runtime_error("reference_point dimensionality mismatch.");
        }
        if (n == 0) {
            throw std::runtime_error("F_combined cannot be empty.");
        }
        auto [fronts, ranks] = fast_non_dominated_sort_impl(F_combined.data(), n, n_obj);
        int64_t worst_rank = 0;
        for (int64_t r : ranks) {
            worst_rank = std::max(worst_rank, r);
        }
        std::vector<size_t> worst;
        for (size_t i = 0; i < n; ++i) {
            if (ranks[i] == worst_rank) {
                worst.push_back(i);
            }
        }
        if (worst.size() == 1) {
            return static_cast<int64_t>(worst[0]);
        }

        std::vector<double> subset;
        subset.reserve(worst.size() * n_obj);
        std::vector<double> ref(reference_point.data(), reference_point.data() + n_obj);
        for (size_t idx : worst) {
            for (size_t m = 0; m < n_obj; ++m) {
                const double v = F_combined.data()[idx * n_obj + m];
                subset.push_back(v);
                ref[m] = std::max(ref[m], v + 1.0e-9);
            }
        }
        const auto contrib = hypervolume_contributions_impl(subset.data(), worst.size(), n_obj, ref.data());
        size_t argmin = 0;
        double best = contrib[0];
        for (size_t i = 1; i < contrib.size(); ++i) {
            if (contrib[i] < best) {
                best = contrib[i];
                argmin = i;
            }
        }
        return static_cast<int64_t>(worst[argmin]);
    });

    m.def(
        "generate_offspring",
        [](NDArray2dDConst X, NDArray2dDConst F, int n_offspring, NDArray1dDConst xl, NDArray1dDConst xu, nb::dict config, uint64_t seed, nb::object out_obj) -> nb::object {
            const size_t n = X.shape(0);
            const size_t n_var = X.shape(1);
            const size_t n_obj = F.shape(1);
            if (F.shape(0) != n) {
                throw std::runtime_error("X/F row mismatch in generate_offspring.");
            }
            if (n_offspring < 0) {
                throw std::runtime_error("n_offspring must be non-negative in generate_offspring.");
            }
            const size_t n_out = static_cast<size_t>(n_offspring);

            auto get_d = [&](const char* key, double def) -> double {
                if (config.contains(key)) {
                    return nb::cast<double>(config[key]);
                }
                return def;
            };
            auto get_i = [&](const char* key, int def) -> int {
                if (config.contains(key)) {
                    return nb::cast<int>(config[key]);
                }
                return def;
            };

            const int pressure = get_i("tournament_pressure", 2);
            const double sbx_prob = get_d("sbx_prob", 0.9);
            const double sbx_eta = get_d("sbx_eta", 20.0);
            const double pm_prob = get_d("pm_prob", 1.0 / std::max<size_t>(1, n_var));
            const double pm_eta = get_d("pm_eta", 20.0);
            const auto [lower, upper] = normalize_bounds_impl(xl.data(), xl.shape(0), xu.data(), xu.shape(0), n_var);

            std::vector<double> children;
            if (n_out > 0) {
                children = generate_offspring_impl(
                    X.data(),
                    F.data(),
                    n,
                    n_var,
                    n_obj,
                    n_out,
                    lower,
                    upper,
                    pressure,
                    sbx_prob,
                    sbx_eta,
                    pm_prob,
                    pm_eta,
                    seed
                );
            } else {
                children.clear();
            }

            if (!out_obj.is_none()) {
                auto out = require_out_float64_c_2d(out_obj, n_out, n_var, "generate_offspring");
                if (children.empty()) {
                    return nb::borrow<nb::object>(out_obj);
                }
                std::copy(children.begin(), children.end(), out.data());
                return nb::borrow<nb::object>(out_obj);
            }

            return make_float64_array_2d(np_module(), children, n_out, n_var);
        },
        nb::arg("X"),
        nb::arg("F"),
        nb::arg("n_offspring"),
        nb::arg("xl"),
        nb::arg("xu"),
        nb::arg("config"),
        nb::arg("seed"),
        nb::arg("out") = nb::none()
    );

    m.def(
        "smsemoa_generate_offspring",
        [](NDArray2dDConst X, NDArray2dDConst F, const std::string& selection, int pressure, NDArray1dDConst xl, NDArray1dDConst xu, nb::dict config, uint64_t seed, nb::object out_obj) -> nb::object {
            const size_t n = X.shape(0);
            const size_t n_var = X.shape(1);
            const size_t n_obj = F.shape(1);
            if (F.shape(0) != n) {
                throw std::runtime_error("X/F row mismatch in smsemoa_generate_offspring.");
            }
            if (n == 0) {
                throw std::runtime_error("X cannot be empty in smsemoa_generate_offspring.");
            }

            auto get_d = [&](const char* key, double def) -> double {
                if (config.contains(key)) {
                    return nb::cast<double>(config[key]);
                }
                return def;
            };
            const double sbx_prob = get_d("sbx_prob", 0.9);
            const double sbx_eta = get_d("sbx_eta", 20.0);
            const double pm_prob = get_d("pm_prob", 1.0 / std::max<size_t>(1, n_var));
            const double pm_eta = get_d("pm_eta", 20.0);

            const auto [lower, upper] = normalize_bounds_impl(xl.data(), xl.shape(0), xu.data(), xu.shape(0), n_var);
            const auto child = smsemoa_generate_offspring_impl(
                X.data(),
                F.data(),
                n,
                n_var,
                n_obj,
                selection,
                pressure,
                lower,
                upper,
                sbx_prob,
                sbx_eta,
                pm_prob,
                pm_eta,
                seed
            );

            if (!out_obj.is_none()) {
                auto out = require_out_float64_c_2d(out_obj, 1, n_var, "smsemoa_generate_offspring");
                if (!child.empty()) {
                    std::copy(child.begin(), child.end(), out.data());
                }
                return nb::borrow<nb::object>(out_obj);
            }

            return make_float64_array_2d(np_module(), child, 1, n_var);
        },
        nb::arg("X"),
        nb::arg("F"),
        nb::arg("selection"),
        nb::arg("pressure"),
        nb::arg("xl"),
        nb::arg("xu"),
        nb::arg("config"),
        nb::arg("seed"),
        nb::arg("out") = nb::none()
    );

    m.def(
        "spea2_generate_offspring",
        [](NDArray2dDConst X, NDArray2dDConst F, int n_offspring, int k_neighbors, NDArray1dDConst xl, NDArray1dDConst xu, nb::dict config, uint64_t seed, nb::object out_obj) -> nb::object {
            const size_t n = X.shape(0);
            const size_t n_var = X.shape(1);
            const size_t n_obj = F.shape(1);
            if (F.shape(0) != n) {
                throw std::runtime_error("X/F row mismatch in spea2_generate_offspring.");
            }
            if (n_offspring < 0) {
                throw std::runtime_error("n_offspring must be non-negative in spea2_generate_offspring.");
            }
            const size_t n_out = static_cast<size_t>(n_offspring);

            auto get_d = [&](const char* key, double def) -> double {
                if (config.contains(key)) {
                    return nb::cast<double>(config[key]);
                }
                return def;
            };
            const double sbx_prob = get_d("sbx_prob", 0.9);
            const double sbx_eta = get_d("sbx_eta", 20.0);
            const double pm_prob = get_d("pm_prob", 1.0 / std::max<size_t>(1, n_var));
            const double pm_eta = get_d("pm_eta", 20.0);

            const auto [lower, upper] = normalize_bounds_impl(xl.data(), xl.shape(0), xu.data(), xu.shape(0), n_var);
            std::vector<double> offspring;
            if (n_out > 0) {
                offspring = spea2_generate_offspring_impl(
                    X.data(),
                    F.data(),
                    n,
                    n_var,
                    n_obj,
                    n_out,
                    k_neighbors,
                    lower,
                    upper,
                    sbx_prob,
                    sbx_eta,
                    pm_prob,
                    pm_eta,
                    seed
                );
            } else {
                offspring.clear();
            }

            if (!out_obj.is_none()) {
                auto out = require_out_float64_c_2d(out_obj, n_out, n_var, "spea2_generate_offspring");
                if (!offspring.empty()) {
                    std::copy(offspring.begin(), offspring.end(), out.data());
                }
                return nb::borrow<nb::object>(out_obj);
            }

            return make_float64_array_2d(np_module(), offspring, n_out, n_var);
        },
        nb::arg("X"),
        nb::arg("F"),
        nb::arg("n_offspring"),
        nb::arg("k_neighbors"),
        nb::arg("xl"),
        nb::arg("xu"),
        nb::arg("config"),
        nb::arg("seed"),
        nb::arg("out") = nb::none()
    );

    m.def(
        "nsga2_evolve",
        [](NDArray2dDConst X0, NDArray2dDConst F0, NDArray1dDConst xl, NDArray1dDConst xu, nb::dict config, int n_generations, uint64_t seed, nb::callable eval_fn) -> nb::tuple {
            const size_t pop_size = X0.shape(0);
            const size_t n_var = X0.shape(1);
            const size_t n_obj = F0.shape(1);
            if (F0.shape(0) != pop_size) {
                throw std::runtime_error("X0/F0 row mismatch in nsga2_evolve.");
            }
            if (n_generations < 0) {
                throw std::runtime_error("n_generations must be non-negative in nsga2_evolve.");
            }

            auto get_d = [&](const char* key, double def) -> double {
                if (config.contains(key)) {
                    return nb::cast<double>(config[key]);
                }
                return def;
            };
            auto get_i = [&](const char* key, int def) -> int {
                if (config.contains(key)) {
                    return nb::cast<int>(config[key]);
                }
                return def;
            };

            const int pressure = get_i("tournament_pressure", 2);
            const double sbx_prob = get_d("sbx_prob", 0.9);
            const double sbx_eta = get_d("sbx_eta", 20.0);
            const double pm_prob = get_d("pm_prob", 1.0 / std::max<size_t>(1, n_var));
            const double pm_eta = get_d("pm_eta", 20.0);
            const auto [lower, upper] = normalize_bounds_impl(xl.data(), xl.shape(0), xu.data(), xu.shape(0), n_var);

            std::vector<double> X(X0.data(), X0.data() + pop_size * n_var);
            std::vector<double> F(F0.data(), F0.data() + pop_size * n_obj);
            std::mt19937_64 rng(seed);
            nb::object np = np_module();

            for (int g = 0; g < n_generations; ++g) {
                const auto child_seed = static_cast<uint64_t>(rng());
                auto X_off_flat = generate_offspring_impl(
                    X.data(),
                    F.data(),
                    pop_size,
                    n_var,
                    n_obj,
                    pop_size,
                    lower,
                    upper,
                    pressure,
                    sbx_prob,
                    sbx_eta,
                    pm_prob,
                    pm_eta,
                    child_seed
                );
                nb::object X_off_obj = make_float64_array_2d(np, X_off_flat, pop_size, n_var);
                nb::object eval_out = eval_fn(X_off_obj);
                nb::object F_off_obj = extract_eval_objectives(eval_out, np);
                auto F_off = nb::cast<NDArray2dDConst>(F_off_obj);
                if (F_off.shape(0) != pop_size || F_off.shape(1) != n_obj) {
                    throw std::runtime_error("eval_fn objective matrix shape mismatch in nsga2_evolve.");
                }
                std::vector<double> F_off_flat(F_off.data(), F_off.data() + pop_size * n_obj);

                auto [X_new, F_new] = nsga2_survival_flat_impl(
                    X,
                    F,
                    X_off_flat,
                    F_off_flat,
                    n_var,
                    n_obj,
                    pop_size,
                    nullptr
                );
                X.swap(X_new);
                F.swap(F_new);
            }

            return nb::make_tuple(
                make_float64_array_2d(np, X, pop_size, n_var),
                make_float64_array_2d(np, F, pop_size, n_obj)
            );
        },
        nb::arg("X0"),
        nb::arg("F0"),
        nb::arg("xl"),
        nb::arg("xu"),
        nb::arg("config"),
        nb::arg("n_generations"),
        nb::arg("seed"),
        nb::arg("eval_fn")
    );

    m.def("dominance_matrix", [](NDArray2dDConst F) {
        const auto dom = dominance_matrix_impl(F.data(), F.shape(0), F.shape(1));
        std::vector<std::vector<uint8_t>> out(F.shape(0), std::vector<uint8_t>(F.shape(0), 0));
        for (size_t i = 0; i < F.shape(0); ++i) {
            for (size_t j = 0; j < F.shape(0); ++j) {
                out[i][j] = dom[i * F.shape(0) + j];
            }
        }
        return out;
    });

    m.def(
        "spea2_fitness",
        [](NDArray2dDConst F, nb::object dom_obj, nb::object k_obj) {
            const size_t n = F.shape(0);
            const size_t n_obj = F.shape(1);
            std::vector<uint8_t> dom;
            if (dom_obj.is_none()) {
                dom = dominance_matrix_impl(F.data(), n, n_obj);
            } else {
                auto dom_arr = nb::cast<nb::ndarray<const bool, nb::ndim<2>, nb::c_contig>>(dom_obj);
                if (dom_arr.shape(0) != n || dom_arr.shape(1) != n) {
                    throw std::runtime_error("dom shape mismatch.");
                }
                dom.resize(n * n, 0);
                for (size_t i = 0; i < n * n; ++i) {
                    dom[i] = dom_arr.data()[i] ? 1 : 0;
                }
            }
            int k = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(n))));
            if (!k_obj.is_none()) {
                k = nb::cast<int>(k_obj);
            }
            auto [fitness, dist] = spea2_fitness_impl(F.data(), n, n_obj, dom.data(), k);

            std::vector<std::vector<double>> dist2d(n, std::vector<double>(n, 0.0));
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    dist2d[i][j] = dist[i * n + j];
                }
            }
            return nb::make_tuple(fitness, dist2d);
        },
        nb::arg("F"),
        nb::arg("dom") = nb::none(),
        nb::arg("k") = nb::none()
    );

    m.def(
        "spea2_environmental_selection_indices",
        [](NDArray2dDConst F, int archive_size, nb::object k_obj) {
            const size_t n = F.shape(0);
            const size_t n_obj = F.shape(1);
            size_t keep = 0;
            if (archive_size <= 0) {
                keep = 0;
            } else {
                keep = std::min(static_cast<size_t>(archive_size), n);
            }
            int k = 1;
            if (!k_obj.is_none()) {
                k = nb::cast<int>(k_obj);
            }
            return spea2_environmental_selection_indices_impl(F.data(), n, n_obj, keep, k);
        },
        nb::arg("F"),
        nb::arg("archive_size"),
        nb::arg("k") = nb::none()
    );

    m.def(
        "ibea_indicator_matrix",
        [](NDArray2dDConst F, NDArray1dDConst reference_point, const std::string& kind) {
            const size_t n = F.shape(0);
            const size_t n_obj = F.shape(1);

            std::vector<double> ref(reference_point.data(), reference_point.data() + reference_point.shape(0));
            if (ref.size() != n_obj) {
                ref.assign(n_obj, 0.0);
                for (size_t m = 0; m < n_obj; ++m) {
                    double mx = -std::numeric_limits<double>::infinity();
                    for (size_t i = 0; i < n; ++i) {
                        mx = std::max(mx, F.data()[i * n_obj + m]);
                    }
                    ref[m] = mx + 1.0;
                }
            }

            std::vector<double> flat;
            if (kind == "hypervolume") {
                flat = hv_indicator_impl(F.data(), n, n_obj, ref.data());
            } else {
                flat = epsilon_indicator_impl(F.data(), n, n_obj);
            }
            std::vector<std::vector<double>> out(n, std::vector<double>(n, 0.0));
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    out[i][j] = flat[i * n + j];
                }
            }
            return out;
        },
        nb::arg("F"),
        nb::arg("reference_point"),
        nb::arg("kind") = "epsilon"
    );

    m.def(
        "ibea_environmental_selection_indices",
        [](NDArray2dDConst F, int pop_size, nb::object reference_point_obj, const std::string& kind, double kappa) {
            const size_t n = F.shape(0);
            const size_t n_obj = F.shape(1);
            size_t keep = 0;
            if (pop_size <= 0) {
                keep = 0;
            } else {
                keep = std::min(static_cast<size_t>(pop_size), n);
            }

            std::vector<double> ref(n_obj, 0.0);
            if (!reference_point_obj.is_none()) {
                auto ref_arr = nb::cast<NDArray1dDConst>(reference_point_obj);
                if (ref_arr.shape(0) != n_obj) {
                    throw std::runtime_error("reference_point dimensionality mismatch.");
                }
                for (size_t m = 0; m < n_obj; ++m) {
                    ref[m] = ref_arr.data()[m];
                }
            } else {
                for (size_t m = 0; m < n_obj; ++m) {
                    double mx = -std::numeric_limits<double>::infinity();
                    for (size_t i = 0; i < n; ++i) {
                        mx = std::max(mx, F.data()[i * n_obj + m]);
                    }
                    ref[m] = mx + 1.0;
                }
            }

            auto [selected, fitness] = ibea_environmental_selection_indices_impl(
                F.data(),
                n,
                n_obj,
                keep,
                ref.data(),
                kind,
                kappa
            );
            return nb::make_tuple(selected, fitness);
        },
        nb::arg("F"),
        nb::arg("pop_size"),
        nb::arg("reference_point") = nb::none(),
        nb::arg("kind") = "epsilon",
        nb::arg("kappa") = 1.0
    );
}
