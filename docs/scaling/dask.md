# Scaling with Dask

VAMOS supports distributed evaluation using Dask for expensive objective functions.

## Installation

```bash
pip install -e ".[distributed]"
```

## Quick Start

```python
from dask.distributed import Client, LocalCluster
import vamos

# Create local cluster
cluster = LocalCluster(n_workers=4)
client = Client(cluster)

# Run optimization
result = vamos.optimize("zdt1", budget=10000)
```

## Connecting to Existing Cluster

```python
from dask.distributed import Client

# Connect to scheduler
client = Client("scheduler.example.com:8786")

# VAMOS will use available workers
result = vamos.optimize("expensive_problem", budget=50000)
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
