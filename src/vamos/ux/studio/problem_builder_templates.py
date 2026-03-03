"""Starter templates for the Studio problem builder."""

from __future__ import annotations

import textwrap

_DEFAULT_TEMPLATE = "ZDT1-like (convex)"


def example_objectives() -> dict[str, dict[str, str]]:
    return {
        **_math_templates(),
        **_domain_templates(),
        "Blank (write your own)": {
            "code": textwrap.dedent("""\
                f0 = x[0]
                f1 = x[1]
                return [f0, f1]"""),
            "n_var": "2",
            "n_obj": "2",
            "category": "blank",
            "description": "Start from scratch with a minimal template. Define your own objectives using Python code.",
            "difficulty": "advanced",
        },
    }


def _math_templates() -> dict[str, dict[str, str]]:
    return {
        "ZDT1-like (convex)": {
            "code": textwrap.dedent("""\
                f0 = x[0]
                g  = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
                f1 = g * (1.0 - (x[0] / g) ** 0.5)
                return [f0, f1]"""),
            "n_var": "5",
            "n_obj": "2",
            "category": "math",
            "description": "A classic two-objective benchmark with a convex trade-off curve. Great starting point to see how optimizers find balanced solutions.",
            "difficulty": "beginner",
        },
        "Schaffer N.1 (concave)": {
            "code": textwrap.dedent("""\
                f0 = x[0] ** 2
                f1 = (x[0] - 2) ** 2
                return [f0, f1]"""),
            "n_var": "1",
            "n_obj": "2",
            "category": "math",
            "description": "The simplest multi-objective problem: one variable, two objectives. Perfect for understanding what a Pareto front looks like.",
            "difficulty": "beginner",
        },
        "Fonseca-Fleming": {
            "code": textwrap.dedent("""\
                import math
                n = len(x)
                s1 = sum((xi - 1.0 / n ** 0.5) ** 2 for xi in x)
                s2 = sum((xi + 1.0 / n ** 0.5) ** 2 for xi in x)
                f0 = 1.0 - math.exp(-s1)
                f1 = 1.0 - math.exp(-s2)
                return [f0, f1]"""),
            "n_var": "3",
            "n_obj": "2",
            "category": "math",
            "description": "A smooth non-linear benchmark where both objectives conflict symmetrically. Tests how well the optimizer explores the full trade-off.",
            "difficulty": "intermediate",
        },
        "Tri-objective (DTLZ1-like)": {
            "code": textwrap.dedent("""\
                g = 1.0 + sum(((xi - 0.5) ** 2) for xi in x[2:])
                f0 = 0.5 * x[0] * x[1] * (1 + g)
                f1 = 0.5 * x[0] * (1 - x[1]) * (1 + g)
                f2 = 0.5 * (1 - x[0]) * (1 + g)
                return [f0, f1, f2]"""),
            "n_var": "5",
            "n_obj": "3",
            "category": "math",
            "description": "A three-objective problem that produces a triangular Pareto surface. Good for exploring many-objective optimization.",
            "difficulty": "advanced",
        },
    }


def _domain_templates() -> dict[str, dict[str, str]]:
    return {
        "Engineering: beam design (cost vs deflection)": {
            "code": textwrap.dedent("""\
                # x[0]=width, x[1]=height
                area = x[0] * x[1]
                cost = 2.0 * x[0] + 3.0 * x[1]
                deflection = 1000.0 / (x[0] * x[1] ** 3 + 1e-6)
                return [cost, deflection]"""),
            "n_var": "2",
            "n_obj": "2",
            "bounds": "1.0, 10.0",
            "category": "engineering",
            "constraint_code": textwrap.dedent("""\
                # Stress must not exceed limit: stress <= 100
                stress = 600.0 / (x[0] * x[1] ** 2 + 1e-6)
                return [stress - 100.0]"""),
            "n_constraints": "1",
            "description": "Design a structural beam: minimize material cost while keeping deflection low, with a stress safety constraint.",
            "difficulty": "intermediate",
        },
        "ML: accuracy vs model size": {
            "code": textwrap.dedent("""\
                import math
                params = x[0] * (x[1] ** 2) * 1000
                acc_proxy = 1.0 - math.exp(-0.5 * x[0] * x[1]) * (1.0 + 0.3 * x[2])
                neg_accuracy = -acc_proxy
                model_size = params / 1e6
                return [neg_accuracy, model_size]"""),
            "n_var": "3",
            "n_obj": "2",
            "bounds": "1.0, 10.0\n1.0, 8.0\n0.0, 0.5",
            "category": "ml",
            "description": "Find the best neural network architecture: maximize accuracy while keeping the model small and efficient.",
            "difficulty": "intermediate",
        },
        "Scheduling: makespan vs tardiness": {
            "code": textwrap.dedent("""\
                n = len(x)
                order = sorted(range(n), key=lambda i: -x[i])
                durations = [2 + i % 3 for i in range(n)]
                deadlines = [3 + 2 * i for i in range(n)]
                t = 0.0
                total_tardiness = 0.0
                for j in order:
                    t += durations[j]
                    total_tardiness += max(0.0, t - deadlines[j])
                makespan = t
                return [makespan, total_tardiness]"""),
            "n_var": "5",
            "n_obj": "2",
            "bounds": "0.0, 1.0",
            "category": "scheduling",
            "description": "Schedule jobs to finish as fast as possible while meeting deadlines. A real-world operations research problem.",
            "difficulty": "advanced",
        },
    }


__all__ = ["_DEFAULT_TEMPLATE", "example_objectives"]
