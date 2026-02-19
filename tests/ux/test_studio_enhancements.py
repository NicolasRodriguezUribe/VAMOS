"""Tests for Studio enhancements: first-launch walkthrough, domain templates,
constraint builder, accessibility CSS, and keyboard shortcuts."""

from __future__ import annotations

from unittest.mock import MagicMock

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
        fn = compile_objective_function(tpl["code"])
        x = np.array([5.0, 5.0])
        result = fn(x)
        assert len(result) == int(tpl["n_obj"])

    def test_ml_template_compiles(self) -> None:
        from vamos.ux.studio.problem_builder_backend import (
            compile_objective_function,
            example_objectives,
        )

        tpl = example_objectives()["ML: accuracy vs model size"]
        fn = compile_objective_function(tpl["code"])
        x = np.array([5.0, 4.0, 0.2])
        result = fn(x)
        assert len(result) == int(tpl["n_obj"])

    def test_scheduling_template_compiles(self) -> None:
        from vamos.ux.studio.problem_builder_backend import (
            compile_objective_function,
            example_objectives,
        )

        tpl = example_objectives()["Scheduling: makespan vs tardiness"]
        fn = compile_objective_function(tpl["code"])
        x = np.random.rand(int(tpl["n_var"]))
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

        fn = compile_constraint_function("return [x[0] + x[1] - 1.0]")
        result = fn(np.array([0.3, 0.5]))
        assert len(result) == 1
        assert abs(result[0] - (-0.2)) < 1e-9

    def test_constraint_syntax_error(self) -> None:
        from vamos.ux.studio.problem_builder_backend import compile_constraint_function

        with pytest.raises(SyntaxError):
            compile_constraint_function("return [x[0] +]")

    def test_engineering_constraint_compiles(self) -> None:
        from vamos.ux.studio.problem_builder_backend import (
            compile_constraint_function,
            example_objectives,
        )

        tpl = example_objectives()["Engineering: beam design (cost vs deflection)"]
        g_code = tpl.get("constraint_code", "")
        assert g_code, "Engineering template should have constraint_code"
        fn = compile_constraint_function(g_code)
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

        fn = compile_objective_function("return [x[0], x[1]]")
        g = compile_constraint_function("return [1.0 - x[0] - x[1]]")
        result = run_preview_optimization(
            fn, n_var=2, n_obj=2, bounds=[(0, 2), (0, 2)],
            algorithm="nsgaii", budget=500, pop_size=20, seed=42,
            constraints=g, n_constraints=1,
        )
        assert result["F"].shape[1] == 2


class TestGenerateScriptWithConstraints:
    """Test script generation with constraints."""

    def test_script_includes_constraints(self) -> None:
        from vamos.ux.studio.problem_builder_backend import generate_script

        script = generate_script(
            "return [x[0], x[1]]",
            name="test", n_var=2, n_obj=2,
            bounds=[(0, 1), (0, 1)], algorithm="nsgaii", budget=3000,
            constraint_code="return [1.0 - x[0] - x[1]]", n_constraints=1,
        )
        assert "constraints" in script
        assert "n_constraints" in script
        compile(script, "<test>", "exec")  # must be valid Python

    def test_script_without_constraints(self) -> None:
        from vamos.ux.studio.problem_builder_backend import generate_script

        script = generate_script(
            "return [x[0], x[1]]",
            name="test", n_var=2, n_obj=2,
            bounds=[(0, 1), (0, 1)], algorithm="nsgaii", budget=3000,
        )
        assert "n_constraints" not in script
        compile(script, "<test>", "exec")


# ======================================================================
# First-launch walkthrough
# ======================================================================


class TestFirstLaunchWalkthrough:
    """Test the first-launch detection and walkthrough rendering."""

    def test_walkthrough_shown_on_first_visit(self) -> None:
        from vamos.ux.studio.app import _render_first_launch_walkthrough

        st = MagicMock()
        st.session_state = {}
        st.button.return_value = False
        _render_first_launch_walkthrough(st)
        st.markdown.assert_called_once()
        html = st.markdown.call_args[0][0]
        assert "First time here" in html

    def test_walkthrough_hidden_after_dismiss(self) -> None:
        from vamos.ux.studio.app import _render_first_launch_walkthrough

        st = MagicMock()
        st.session_state = {"walkthrough_dismissed": True}
        _render_first_launch_walkthrough(st)
        st.markdown.assert_not_called()


# ======================================================================
# Accessibility CSS
# ======================================================================


class TestAccessibilityCSS:
    """Test that custom CSS is injected."""

    def test_css_injected(self) -> None:
        from vamos.ux.studio.app import _inject_accessibility_css

        st = MagicMock()
        _inject_accessibility_css(st)
        st.markdown.assert_called_once()
        css = st.markdown.call_args[0][0]
        assert "focus-visible" in css
        assert "sr-only" in css
        assert "kbd-hint" in css
