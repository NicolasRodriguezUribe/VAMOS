"""
Registry for variation operators.
"""

from __future__ import annotations

from vamos.foundation.registry import Registry

# Global registry for variation operators (classes or factories).
# Key: operator name (e.g. "sbx", "pm").
# Value: Class type or factory function.
_operator_registry: Registry[object] | None = None


def _get_registry() -> Registry[object]:
    global _operator_registry
    if _operator_registry is not None:
        return _operator_registry

    reg: Registry[object] = Registry("VariationOperators")

    # Register common operators
    from vamos.operators.impl.real import (
        ArithmeticCrossover,
        BLXAlphaCrossover,
        CauchyMutation,
        ClampRepair,
        DEMatingCrossover,
        GaussianMutation,
        LinkedPolynomialMutation,
        NonUniformMutation,
        PCXCrossover,
        PolynomialMutation,
        ReflectRepair,
        ResampleRepair,
        RoundRepair,
        SBXCrossover,
        SPXCrossover,
        UNDXCrossover,
        UniformMutation,
        UniformResetMutation,
    )

    # Crossover
    reg.register("sbx", SBXCrossover)
    reg.register("blx_alpha", BLXAlphaCrossover)
    reg.register("arithmetic", ArithmeticCrossover)
    reg.register("de", DEMatingCrossover)
    reg.register("pcx", PCXCrossover)
    reg.register("undx", UNDXCrossover)
    reg.register("simplex", SPXCrossover)

    # Mutation
    reg.register("pm", PolynomialMutation)
    reg.register("polynomial", PolynomialMutation)
    reg.register("non_uniform", NonUniformMutation)
    reg.register("gaussian", GaussianMutation)
    reg.register("uniform_reset", UniformResetMutation)
    reg.register("cauchy", CauchyMutation)
    reg.register("uniform", UniformMutation)
    reg.register("linked_polynomial", LinkedPolynomialMutation)

    # Repair
    reg.register("clip", ClampRepair)
    reg.register("clamp", ClampRepair)
    reg.register("reflect", ReflectRepair)
    reg.register("random", ResampleRepair)
    reg.register("resample", ResampleRepair)
    reg.register("round", RoundRepair)

    # Binary operators
    from vamos.operators.impl.binary import (
        BitFlipMutation,
        HuxCrossover,
        OnePointCrossover,
        TwoPointCrossover,
    )
    from vamos.operators.impl.binary import (
        UniformCrossover as BinaryUniformCrossover,
    )

    reg.register("bitflip", BitFlipMutation)
    reg.register("one_point", OnePointCrossover)
    reg.register("two_point", TwoPointCrossover)
    reg.register("binary_uniform", BinaryUniformCrossover)
    reg.register("hux", HuxCrossover)
    reg.register("spx", SPXCrossover)

    # Permutation operators
    from vamos.operators.impl.permutation import (
        CycleCrossover,
        DisplacementMutation,
        EdgeRecombinationCrossover,
        InsertMutation,
        InversionMutation,
        OrderCrossover,
        PMXCrossover,
        PositionBasedCrossover,
        ScrambleMutation,
        SwapMutation,
    )

    reg.register("swap", SwapMutation)
    reg.register("pmx", PMXCrossover)
    reg.register("cx", CycleCrossover)
    reg.register("cycle", CycleCrossover)
    reg.register("position_based", PositionBasedCrossover)
    reg.register("erx", EdgeRecombinationCrossover)
    reg.register("ox", OrderCrossover)
    reg.register("order", OrderCrossover)
    reg.register("insert", InsertMutation)
    reg.register("scramble", ScrambleMutation)
    reg.register("inversion", InversionMutation)
    reg.register("displacement", DisplacementMutation)

    # Integer operators
    from vamos.operators.impl.integer import (
        ArithmeticIntegerCrossover,
        CreepMutation,
        IntegerPolynomialMutation,
        IntegerSBXCrossover,
        RandomResetMutation,
        UniformIntegerCrossover,
    )

    reg.register("int_uniform", UniformIntegerCrossover)
    reg.register("int_arithmetic", ArithmeticIntegerCrossover)
    reg.register("int_sbx", IntegerSBXCrossover)
    reg.register("reset", RandomResetMutation)
    reg.register("int_pm", IntegerPolynomialMutation)
    reg.register("creep", CreepMutation)

    # Mixed operators
    from vamos.operators.impl.mixed import MixedCrossover, MixedMutation

    reg.register("mixed", MixedCrossover)
    reg.register("mixed_mutation", MixedMutation)

    _operator_registry = reg
    return reg


def get_operator_registry() -> Registry[object]:
    """Return the global variation operator registry (lazily initialized)."""
    return _get_registry()


def __getattr__(name: str) -> object:
    if name == "operator_registry":
        return _get_registry()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
