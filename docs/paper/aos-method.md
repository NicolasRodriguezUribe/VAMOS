# Adaptive Operator Selection (AOS) for NSGA-II

This note documents the public AOS contract in VAMOS. It covers the operator
portfolio, supported policies, reward definitions, and the logging schema.

## Operator portfolio arms

An AOS "arm" corresponds to a single variation pipeline (crossover + mutation)
defined in `adaptive_operator_selection.operator_pool`. Each arm is assigned:

- `op_id`: stable index in the pool (string form of the index, e.g., "0").
- `op_name`: `<crossover>+<mutation>` for the pipeline.

NSGA-II selects exactly one arm per generation when AOS is enabled.

## Policies

Supported methods in the NSGA-II config contract
(`adaptive_operator_selection.method`):

- `epsilon_greedy`
- `ucb`
- `exp3`
- `thompson_sampling`
- `sliding_ucb` (requires `window_size > 0`)

## Reward definitions

The MVP reward uses survival credit:

- `reward_survival = survivors_from_last_offspring / offspring_count`

Optional ND insertion reward (proxy):

- `reward_nd_insertions = nondominated_offspring_survivors / offspring_count`

This proxy counts offspring that survive into the next population and are
non-dominated in that next population. It is intentionally index-based, not
based on floating equality of decision vectors.

`reward_hv_delta` is reserved and currently recorded as 0.0.

## Credit timing

Rewards are computed **once per generation**, after survival selection is
complete. The policy is updated once per generation with the aggregated reward.

## Reproducibility

Determinism is controlled by:

- `seed` (global run seed)
- `adaptive_operator_selection.rng_seed` (policy RNG)

## Exploration floor

`adaptive_operator_selection.floor_prob` mixes a uniform draw into selection
with probability `floor_prob`. Set it to 0 to disable; 1.0 makes selection
fully uniform.

With the same engine, configuration, and seeds, operator selection and rewards
are deterministic.

## Logging contract (CSV schemas)

Files are written only when AOS is enabled and trace data is present.

`aos_trace.csv` columns:

```
step,mating_id,op_id,op_name,reward,reward_survival,reward_nd_insertions,reward_hv_delta,batch_size
```

`aos_summary.csv` columns:

```
op_id,op_name,pulls,mean_reward,total_reward,usage_fraction
```

## Minimal run command

Example config (canonical):

- `examples/configs/nsgaii_aos_min.yml`

Run:

```
.\.venv\Scripts\python.exe -m vamos.experiment.cli.main --config examples/configs/nsgaii_aos_min.yml
```

Output directory pattern:

```
results/aos_example/<PROBLEM>/<algorithm>/<engine>/seed_<seed>/
```
