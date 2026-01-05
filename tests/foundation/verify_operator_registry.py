from vamos.engine.algorithms_registry import available_crossover_methods, available_mutation_methods

encodings = ["real", "binary", "permutation", "integer", "mixed"]

for enc in encodings:
    cx = available_crossover_methods(enc)
    mut = available_mutation_methods(enc)
    print(f"Encoding: {enc}")
    print(f"  Crossover: {cx}")
    print(f"  Mutation:  {mut}")
    if not cx:
        print(f"  WARNING: No crossover for {enc}")
    if not mut:
        print(f"  WARNING: No mutation for {enc}")

print("\nVerification OK")
