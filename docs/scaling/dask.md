# Scaling with Dask

VAMOS supports distributed evaluation using Dask for expensive objective functions.

## Installation

```bash
pip install -e ".[compute]"
```

## Quick Start

```python
from dask.distributed import Client, LocalCluster
from vamos.foundation.eval.backends import DaskEvalBackend
from vamos import make_problem_selection, optimize
from vamos.algorithms import NSGAIIConfig

# Create local cluster
cluster = LocalCluster(n_workers=4)
client = Client(cluster)

backend = DaskEvalBackend(client=client)

# Run optimization (distributed evaluation)
problem = make_problem_selection("zdt1", n_var=30).instantiate()
algo_cfg = NSGAIIConfig.default(pop_size=100, n_var=problem.n_var)
result = optimize(
    problem,
    algorithm="nsgaii",
    algorithm_config=algo_cfg,
    termination=("max_evaluations", 10_000),
    seed=42,
    engine="numpy",
    eval_strategy=backend,
)
```

## Connecting to Existing Cluster

```python
from vamos import make_problem_selection, optimize
from vamos.algorithms import NSGAIIConfig
from vamos.foundation.eval.backends import DaskEvalBackend

backend = DaskEvalBackend(address="scheduler.example.com:8786")
problem = make_problem_selection("zdt1", n_var=30).instantiate()
algo_cfg = NSGAIIConfig.default(pop_size=100, n_var=problem.n_var)
result = optimize(
    problem,
    algorithm="nsgaii",
    algorithm_config=algo_cfg,
    termination=("max_evaluations", 50_000),
    seed=42,
    engine="numpy",
    eval_strategy=backend,
)
```

## Kubernetes Deployment

```yaml
# dask-cluster.yaml
apiVersion: kubernetes.dask.org/v1
kind: DaskCluster
metadata:
  name: vamos-cluster
spec:
  worker:
    replicas: 10
    resources:
      limits:
        memory: "4Gi"
        cpu: "2"
```

## When to Use Distributed

| Scenario | Recommended |
|----------|-------------|
| Cheap objectives (<1ms) | No |
| Medium objectives (10-100ms) | Maybe |
| Expensive objectives (>1s) | Yes |
| Large populations (>1000) | Yes |

## Example

See `examples/distributed/dask_cluster.py` for a complete example.

```bash
# Local test
python examples/distributed/dask_cluster.py --compare

# Connect to cluster
python examples/distributed/dask_cluster.py --scheduler scheduler:8786
```
