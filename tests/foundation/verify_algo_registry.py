from vamos.engine.algorithms_registry import available_algorithms, resolve_algorithm

print("Algorithms:", available_algorithms())

try:
    builder = resolve_algorithm("nsgaii")
    print("Resolved NSGAII:", builder)
except Exception as e:
    print("Failed to resolve NSGAII:", e)
    exit(1)
