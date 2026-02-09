"""Tests for the VAMOS Studio Problem Builder backend logic."""

from __future__ import annotations

import pytest
import numpy as np

from vamos.ux.studio.problem_builder_backend import (
    compile_objective_function,
    example_objectives,
    generate_script,
    parse_bounds_text,
    run_preview_optimization,
)


# ======================================================================
# compile_objective_function
# ======================================================================


class TestCompileObjectiveFunction:
    """Test the user-code compilation helper."""

    def test_simple_function(self) -> None:
        fn = compile_objective_function("return [x[0], 1 - x[0]]")
        result = fn(np.array([0.3, 0.5]))
        assert len(result) == 2
        assert abs(result[0] - 0.3) < 1e-9
        assert abs(result[1] - 0.7) < 1e-9

    def test_multiline_with_math(self) -> None:
        code = "import math\nf0 = x[0] ** 2\nf1 = math.sqrt(x[0])\nreturn [f0, f1]"
        fn = compile_objective_function(code)
        result = fn(np.array([4.0]))
        assert abs(result[0] - 16.0) < 1e-9
        assert abs(result[1] - 2.0) < 1e-9

    def test_numpy_available(self) -> None:
        code = "return [float(np.sum(x)), float(np.prod(x))]"
        fn = compile_objective_function(code)
        result = fn(np.array([2.0, 3.0]))
        assert abs(result[0] - 5.0) < 1e-9
        assert abs(result[1] - 6.0) < 1e-9

    def test_syntax_error(self) -> None:
        with pytest.raises(SyntaxError):
            compile_objective_function("return [x[0] +]")

    def test_all_templates_compile(self) -> None:
        """Every built-in template must compile without error."""
        for name, template in example_objectives().items():
            fn = compile_objective_function(template["code"])
            n_var = int(template["n_var"])
            x = np.random.rand(n_var)
            result = fn(x)
            assert isinstance(result, list), f"Template '{name}' must return a list"
            assert len(result) == int(template["n_obj"]), f"Template '{name}' objective count mismatch"


# ======================================================================
# _parse_bounds_text
# ======================================================================


class TestParseBoundsText:
    """Test the bounds textarea parser."""

    def test_single_line(self) -> None:
        result = parse_bounds_text("0.0, 1.0", 3)
        assert result == [(0.0, 1.0)] * 3

    def test_multi_line(self) -> None:
        text = "0.0, 1.0\n-5.0, 5.0"
        result = parse_bounds_text(text, 2)
        assert result == [(0.0, 1.0), (-5.0, 5.0)]

    def test_with_parens(self) -> None:
        result = parse_bounds_text("(0, 10)", 2)
        assert result == [(0.0, 10.0)] * 2

    def test_error_wrong_count(self) -> None:
        result = parse_bounds_text("0,1\n0,1\n0,1", 2)
        assert isinstance(result, str)
        assert "Expected" in result

    def test_error_inverted(self) -> None:
        result = parse_bounds_text("5.0, 0.0", 1)
        assert isinstance(result, str)
        assert "lower" in result.lower() or "Lower" in result

    def test_error_not_numbers(self) -> None:
        result = parse_bounds_text("a, b", 1)
        assert isinstance(result, str)


# ======================================================================
# _generate_script
# ======================================================================


class TestGenerateScript:
    """Test the export script generator."""

    def test_valid_python(self) -> None:
        script = generate_script(
            "return [x[0], 1 - x[0]]",
            name="test_problem",
            n_var=2,
            n_obj=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            algorithm="nsgaii",
            budget=5000,
        )
        # Must be valid Python
        compile(script, "<test>", "exec")

    def test_contains_make_problem(self) -> None:
        script = generate_script(
            "return [x[0]]",
            name="test",
            n_var=1,
            n_obj=1,
            bounds=[(0.0, 1.0)],
            algorithm="nsgaii",
            budget=3000,
        )
        assert "make_problem" in script
        assert "optimize" in script
        assert "max_evaluations=3000" in script

    def test_contains_problem_name(self) -> None:
        script = generate_script(
            "return [x[0]]",
            name="my cool problem",
            n_var=1,
            n_obj=1,
            bounds=[(0.0, 1.0)],
            algorithm="moead",
            budget=5000,
        )
        assert "my cool problem" in script
        assert 'algorithm="moead"' in script


# ======================================================================
# run_preview_optimization (integration)
# ======================================================================


class TestRunPreviewOptimization:
    """Integration test: compile + optimize end-to-end."""

    def test_basic_preview(self) -> None:
        fn = compile_objective_function("return [x[0], 1 - x[0]]")
        result = run_preview_optimization(
            fn,
            n_var=2,
            n_obj=2,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            algorithm="nsgaii",
            budget=500,
            pop_size=20,
            seed=42,
        )
        assert "F" in result
        assert result["F"].shape[1] == 2
        assert result["F"].shape[0] > 0
        assert result["elapsed_ms"] > 0

    def test_three_objective(self) -> None:
        code = "return [x[0], x[1], 1 - x[0] - x[1]]"
        fn = compile_objective_function(code)
        result = run_preview_optimization(
            fn,
            n_var=3,
            n_obj=3,
            bounds=[(0.0, 1.0)] * 3,
            algorithm="nsgaii",
            budget=500,
            pop_size=20,
            seed=0,
        )
        assert result["F"].shape[1] == 3
