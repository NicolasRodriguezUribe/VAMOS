from vamos.algorithms import available_crossover_methods, available_mutation_methods


def test_available_crossover_methods_real():
    methods = available_crossover_methods("real")
    assert isinstance(methods, tuple)
    assert len(methods) > 0
    # Check for known real crossovers
    assert "sbx" in methods
    assert "blx_alpha" in methods
    # Should not contain binary/perm operators
    assert "one_point" not in methods
    assert "pmx" not in methods


def test_available_mutation_methods_real():
    methods = available_mutation_methods("real")
    assert isinstance(methods, tuple)
    assert len(methods) > 0
    # Check for known real mutations
    assert "pm" in methods
    assert "gaussian" in methods
    # Should not contain binary/perm operators
    assert "bit_flip" not in methods
    assert "scramble" not in methods


def test_available_crossover_methods_binary():
    methods = available_crossover_methods("binary")
    assert "one_point" in methods
    assert "two_point" in methods
    assert "sbx" not in methods


def test_available_mutation_methods_binary():
    methods = available_mutation_methods("binary")
    assert "bit_flip" in methods
    assert "pm" not in methods


def test_available_crossover_methods_permutation():
    methods = available_crossover_methods("permutation")
    assert "pmx" in methods
    assert "order" in methods
    assert "sbx" not in methods


def test_available_crossover_methods_default():
    # Should default to "real"
    assert available_crossover_methods() == available_crossover_methods("real")


def test_invalid_encoding():
    # For now it just returns empty tuple or raises keyerror depending on implementation
    # Implementation is a simple encoding switch over operator registries.
    # Let's check what happens for invalid encoding.
    # It attempts key access on dictionaries for other encodings.
    # Wait, the implementation was:
    # if encoding == "real": return tuple(sorted(REAL_CROSSOVER))
    # elif encoding == "permutation": return tuple(sorted(PERM_CROSSOVER.keys()))
    # ...
    # else: return ()
    # So it should return empty tuple for unknown encoding.

    assert available_crossover_methods("unknown") == ()
