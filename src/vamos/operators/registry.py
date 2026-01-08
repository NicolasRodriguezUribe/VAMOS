"""
Registry for variation operators.
"""

from __future__ import annotations

from vamos.foundation.registry import Registry

# Global registry for variation operators (classes or factories)
# Key: operator name (e.g. "sbx", "pm")
# Value: Class type or factory function
# Global registry for variation operators (classes or factories)
# Key: operator name (e.g. "sbx", "pm")
# Value: Class type or factory function
_operator_registry: Registry | None = None


def _get_registry() -> Registry:
    global _operator_registry
    if _operator_registry is not None:
        return _operator_registry

    reg = Registry("VariationOperators")

    # Register common operators
    from vamos.operators.real import (
        SBXCrossover,
        PolynomialMutation,
        ArithmeticCrossover,
        PCXCrossover,
        SPXCrossover,
        UNDXCrossover,
        BLXAlphaCrossover,
        CauchyMutation,
        GaussianMutation,
        NonUniformMutation,
        LinkedPolynomialMutation,
        UniformMutation,
        UniformResetMutation,
        ClampRepair,
        ReflectRepair,
        ResampleRepair,
        RoundRepair,
    )

    # Crossover
    reg.register("sbx", SBXCrossover)
    reg.register("blx_alpha", BLXAlphaCrossover)
    reg.register("arithmetic", ArithmeticCrossover)
    reg.register("pcx", PCXCrossover)
    reg.register("undx", UNDXCrossover)
    reg.register("spx", SPXCrossover)

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
    from vamos.operators.binary import (
        BitFlipMutation,
        OnePointCrossover,
        TwoPointCrossover,
        UniformCrossover as BinaryUniformCrossover,
        HuxCrossover,
    )

    reg.register("bitflip", BitFlipMutation)
    reg.register("one_point", OnePointCrossover)
    reg.register("two_point", TwoPointCrossover)
    reg.register("binary_uniform", BinaryUniformCrossover)
    reg.register("hux", HuxCrossover)

    # Permutation operators
    from vamos.operators.permutation import (
        SwapMutation,
        PMXCrossover,
        CycleCrossover,
        PositionBasedCrossover,
        EdgeRecombinationCrossover,
        OrderCrossover,
        InsertMutation,
        ScrambleMutation,
        InversionMutation,
        DisplacementMutation,
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
    from vamos.operators.integer import (
        UniformIntegerCrossover,
        ArithmeticIntegerCrossover,
        RandomResetMutation,
        CreepMutation,
    )

    reg.register("int_uniform", UniformIntegerCrossover)
    reg.register("int_arithmetic", ArithmeticIntegerCrossover)
    reg.register("reset", RandomResetMutation)
    reg.register("creep", CreepMutation)

    # Mixed operators
    from vamos.operators.mixed import MixedCrossover, MixedMutation

    reg.register("mixed", MixedCrossover)
    reg.register("mixed_mutation", MixedMutation)

    _operator_registry = reg
    return reg


def __getattr__(name: str):
    if name == "operator_registry":
        return _get_registry()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
