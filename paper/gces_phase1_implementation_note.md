# GCES Phase-1 Implementation Note

This note records the exact current behavior of the `gces` MVP implementation.
It is implementation-aligned and intentionally narrower than a full algorithm
specification.

## Scope and Host

- Public algorithm key: `gces`
- Internal host loop: NSGA-II
- Divergence from NSGA-II: split-front truncation only
- Unchanged relative to baseline NSGA-II:
  - parent/mating selection
  - crossover and mutation
  - offspring generation
  - non-dominated sorting
  - complete-front filling before the split front

## Current `GCES.tell()` Behavior

For one generational update, `GCES.tell()`:

1. Rejects unsupported phase-1 modes:
   - constrained problems
   - steady-state mode
   - incremental replacement (`offspring_size < pop_size`)
   - genealogy tracking
   - `moocore`
2. Merges parent and offspring populations into `(combined_X, combined_F)`.
3. Computes non-dominated ranks on the merged objective matrix.
4. Fills all complete fronts unchanged until the next front would overflow the target population size.
5. Defines:
   - `ideal = min(combined_F, axis=0)`
   - `nadir = max(combined_F, axis=0)`
6. Calls `select_split_front_gces(F_split, slots, ideal, nadir, rng)` on the split front only.
7. Builds the new population from:
   - all previously accepted complete fronts
   - the selected subset of the split front
8. Updates the archive from the combined non-dominated candidates.
9. Checks HV termination using the standard NSGA-II state machinery.

## Current `select_split_front_gces(...)` Behavior

### Trivial cases

- If `slots >= n_split`, return all local indices.
- If `slots <= 0`, return an empty index array.

### Normalization

- Objectives are min-max normalized using the provided `ideal` and `nadir`.
- For coordinate `m`, if `nadir[m] - ideal[m] == 0`, then the entire normalized
  coordinate is set to `0`.

### Component detection

1. Compute complete-graph Euclidean distances on the normalized split front.
2. Build a deterministic Kruskal MST.
   - Edges are ordered by:
     1. smaller edge weight
     2. smaller first endpoint index
     3. smaller second endpoint index
3. Let `L` be the vector of MST edge lengths.
4. Compute:
   - `med = median(L)`
   - `mad = median(|L - med|)`
5. Cut MST edges:
   - if `mad > 0`: cut edges with length `> med + 3 * mad`
   - if `mad == 0`: cut edges with length `> med`
6. Connected components after edge removal are the GCES components.
7. If no edge satisfies the cut rule, the split front remains a single component.

### Component weights and slot allocation

For component `j`:

- `n_j` = component size
- `D_j` = Euclidean diameter in normalized objective space
- `weight_j = log(1 + n_j) * D_j`

Allocation rule:

1. Every kept component receives at least one slot.
2. If `#components > slots`:
   - keep the top `slots` components by descending `weight_j`
   - tie-break by smaller minimum local index
   - assign one slot to each kept component
3. Otherwise:
   - assign one slot to each component
   - distribute remaining slots by largest remainder on the normalized weights
   - tie-break by smaller minimum local index
4. If the total component weight is zero:
   - distribute remaining slots deterministically by larger component size
   - tie-break by smaller minimum local index

Slots are never allocated beyond the component capacity `n_j`.

## Intra-component selection

### First seed

- The first survivor in each component is the point with minimum Euclidean
  distance to the normalized ideal.
- Tie-break: smaller local index.

### Local graph

For a component of size `n_j`:

- Build a symmetric kNN graph on normalized objectives with
  `k = min(n_j - 1, max(3, min(10, ceil(log2(n_j)))))`
- Add the component MST edges to that graph.
- Edge weights are Euclidean distances in normalized objective space.
- All-pairs geodesic distances are shortest-path distances on that graph.

### Geodesic farthest-first

After the first seed, repeatedly add the point maximizing its minimum geodesic
distance to the already selected set.

Tie-breaks:

1. smaller distance to the normalized ideal
2. smaller original local index in the split front

Returned local indices are sorted in ascending order.

## Paper-Style Pseudocode

### Algorithm-Level `GCES.tell()`

```text
procedure GCES_TELL(state, F_off)
    reject unsupported phase-1 modes

    X_off <- state.pending_offspring
    combined_X <- concatenate(state.X, X_off)
    combined_F <- concatenate(state.F, F_off)

    ranks <- NSGA2_RANKING(combined_F)
    fronts <- FRONTS_FROM_RANKS(ranks)

    ideal <- columnwise_min(combined_F)
    nadir <- columnwise_max(combined_F)
    selected <- []

    for front in fronts do
        if |selected| + |front| <= pop_size then
            append all indices in front to selected
        else
            slots <- pop_size - |selected|
            local_idx <- SELECT_SPLIT_FRONT_GCES(
                combined_F[front], slots, ideal, nadir, rng
            )
            append front[local_idx] to selected
            break
        end if
    end for

    state.X <- combined_X[selected]
    state.F <- combined_F[selected]
    state.G <- None
    state.pending_offspring_ids <- None

    archive_candidates <- nondominated points from (combined_X, combined_F)
    update archive with archive_candidates

    return hv_termination_reached(state)
end procedure
```

### Selector-Level `select_split_front_gces(...)`

```text
procedure SELECT_SPLIT_FRONT_GCES(F_split, slots, ideal, nadir, rng)
    if slots >= n_split then
        return [0, 1, ..., n_split - 1]
    end if
    if slots <= 0 then
        return []
    end if

    normalize F_split by min-max using ideal and nadir
    for any zero-span coordinate, set the normalized coordinate to 0

    D <- complete-graph Euclidean distances on normalized points
    T <- deterministic Kruskal MST of D
    L <- edge lengths of T

    med <- median(L)
    mad <- median(|L - med|)

    if mad > 0 then
        cut edges with length > med + 3 * mad
    else
        cut edges with length > med
    end if

    components <- connected components after cutting

    for each component j do
        n_j <- size(component j)
        D_j <- Euclidean diameter of component j
        weight_j <- log(1 + n_j) * D_j
    end for

    if number_of_components > slots then
        keep the top slots components by descending weight_j
        assign 1 slot to each kept component
    else
        assign 1 slot to every component
        distribute remaining slots by largest remainder on weight_j
        if total weight is zero, distribute deterministically by larger size
    end if

    selected <- []
    for each kept component j with assigned slots s_j do
        first seed <- point closest to normalized ideal
        if s_j > 1 then
            build symmetric kNN graph with
                k = min(n_j - 1, max(3, min(10, ceil(log2(n_j)))))
            add component MST edges
            compute all-pairs shortest-path distances
            repeatedly add the point maximizing minimum geodesic distance
            break ties by:
                1. smaller ideal distance
                2. smaller local index
        end if
        append selected local indices from component j
    end for

    return sort(selected)
end procedure
```

## Implementation-to-Paper Mapping

- “Environmental selection remains NSGA-II until the split front”
  - implemented exactly in `GCES.tell()`
- “GCES operates on the split front only”
  - implemented exactly by `select_split_front_gces(...)`
- “Normalization uses provided bounds”
  - implemented exactly, with `ideal/nadir` supplied from the merged population
- “Components detected by MST cutting”
  - implemented exactly with the median/MAD rule above
- “At least one slot per component”
  - implemented exactly for every kept component
- “Geodesic farthest-first within components”
  - implemented exactly on the kNN ∪ MST graph

## Intentionally Unsupported in Phase 1

- constrained problems
- steady-state mode
- incremental replacement
- `moocore`
- genealogy

## Known Follow-Up Items

- If the paper wants normalization over the split front only, the code currently
  does not do that; it uses the full merged population bounds.
- The implementation currently computes all-pairs shortest paths with a dense
  Floyd-Warshall-style update. This is acceptable for the MVP but should be
  revisited if split fronts become large.
- No ablations are present yet.
