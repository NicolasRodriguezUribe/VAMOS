"""Tests for Studio enhancements: first-launch walkthrough, domain templates,
constraint builder, accessibility CSS, and keyboard shortcuts."""

from __future__ import annotations

import numpy as np
import pytest

# ======================================================================
# Domain templates
# ======================================================================


class TestDomainTemplates:
    """Verify domain-specific templates compile and return correct shapes."""

    def test_all_templates_have_category(self) -> None:
        from vamos.ux.studio.problem_builder_backend import example_objectives

        for name, tpl in example_objectives().items():
            assert "category" in tpl, f"Template '{name}' missing 'category' key"

    def test_engineering_template_compiles(self) -> None:
        from vamos.ux.studio.problem_builder_backend import (
            compile_objective_function,
            example_objectives,
        )

        tpl = example_objectives()["Engineering: beam design (cost vs deflection)"]
        fn = compile_objective_function(tpl["code"], trusted_local_code=True)
        x = np.array([5.0, 5.0])
        result = fn(x)
        assert len(result) == int(tpl["n_obj"])

    def test_ml_template_compiles(self) -> None:
        from vamos.ux.studio.problem_builder_backend import (
            compile_objective_function,
            example_objectives,
        )

        tpl = example_objectives()["ML: accuracy vs model size"]
        fn = compile_objective_function(tpl["code"], trusted_local_code=True)
        x = np.array([5.0, 4.0, 0.2])
        result = fn(x)
        assert len(result) == int(tpl["n_obj"])

    def test_scheduling_template_compiles(self) -> None:
        from vamos.ux.studio.problem_builder_backend import (
            compile_objective_function,
            example_objectives,
        )

        tpl = example_objectives()["Scheduling: makespan vs tardiness"]
        fn = compile_objective_function(tpl["code"], trusted_local_code=True)
        x = np.random.default_rng(0).random(int(tpl["n_var"]))
        result = fn(x)
        assert len(result) == int(tpl["n_obj"])

    def test_domain_filter_categories(self) -> None:
        from vamos.ux.studio.problem_builder_backend import example_objectives

        templates = example_objectives()
        categories = {t.get("category") for t in templates.values()}
        assert "engineering" in categories
        assert "ml" in categories
        assert "scheduling" in categories
        assert "math" in categories


# ======================================================================
# Constraint builder backend
# ======================================================================


class TestConstraintCompilation:
    """Test compile_constraint_function."""

    def test_simple_constraint(self) -> None:
        from vamos.ux.studio.problem_builder_backend import compile_constraint_function

        fn = compile_constraint_function("return [x[0] + x[1] - 1.0]", trusted_local_code=True)
        result = fn(np.array([0.3, 0.5]))
        assert len(result) == 1
        assert abs(result[0] - (-0.2)) < 1e-9

    def test_constraint_syntax_error(self) -> None:
        from vamos.ux.studio.problem_builder_backend import compile_constraint_function

        with pytest.raises(SyntaxError):
            compile_constraint_function("return [x[0] +]", trusted_local_code=True)

    def test_engineering_constraint_compiles(self) -> None:
        from vamos.ux.studio.problem_builder_backend import (
            compile_constraint_function,
            example_objectives,
        )

        tpl = example_objectives()["Engineering: beam design (cost vs deflection)"]
        g_code = tpl.get("constraint_code", "")
        assert g_code, "Engineering template should have constraint_code"
        fn = compile_constraint_function(g_code, trusted_local_code=True)
        result = fn(np.array([5.0, 5.0]))
        assert isinstance(result, list)
        assert len(result) == int(tpl["n_constraints"])


class TestRunPreviewWithConstraints:
    """Integration: preview optimization with constraints."""

    def test_constrained_preview(self) -> None:
        from vamos.ux.studio.problem_builder_backend import (
            compile_constraint_function,
            compile_objective_function,
            run_preview_optimization,
        )

        fn = compile_objective_function("return [x[0], x[1]]", trusted_local_code=True)
        g = compile_constraint_function("return [1.0 - x[0] - x[1]]", trusted_local_code=True)
        result = run_preview_optimization(
            fn,
            n_var=2,
            n_obj=2,
            bounds=[(0, 2), (0, 2)],
            algorithm="nsgaii",
            budget=500,
            pop_size=20,
            seed=42,
            constraints=g,
            n_constraints=1,
            trusted_local_code=True,
        )
        assert result["F"].shape[1] == 2


class TestGenerateScriptWithConstraints:
    """Test script generation with constraints."""

    def test_script_includes_constraints(self) -> None:
        from vamos.ux.studio.problem_builder_backend import generate_script

        script = generate_script(
            "return [x[0], x[1]]",
            name="test",
            n_var=2,
            n_obj=2,
            bounds=[(0, 1), (0, 1)],
            algorithm="nsgaii",
            budget=3000,
            constraint_code="return [1.0 - x[0] - x[1]]",
            n_constraints=1,
        )
        assert "constraints" in script
        assert "n_constraints" in script
        compile(script, "<test>", "exec")  # must be valid Python

    def test_script_without_constraints(self) -> None:
        from vamos.ux.studio.problem_builder_backend import generate_script

        script = generate_script(
            "return [x[0], x[1]]",
            name="test",
            n_var=2,
            n_obj=2,
            bounds=[(0, 1), (0, 1)],
            algorithm="nsgaii",
            budget=3000,
        )
        assert "n_constraints" not in script
        compile(script, "<test>", "exec")
