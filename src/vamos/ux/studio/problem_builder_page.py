"""Main page composition for the Studio problem builder."""

from __future__ import annotations

from typing import Any

from vamos.ux.studio.problem_builder_backend import (
    compile_objective_function,
    example_objectives,
    generate_script,
    parse_bounds_text,
    run_preview_optimization,
)
from vamos.ux.studio.problem_builder_templates import _DEFAULT_TEMPLATE
from vamos.ux.studio.problem_builder_ui import (
    _ALGO_HELP,
    _ALGO_LABELS,
    coerce_bool_widget_value,
    render_builder_intro,
    render_constraint_section,
    render_preview_guidance,
    render_preview_plot,
    render_summary_table,
    render_template_chips,
    render_template_snapshot,
)


def render_problem_builder(st: Any, px: Any) -> None:
    render_builder_intro(st)
    advanced_mode = _resolve_advanced_mode(st)
    template = _select_template(st, advanced_mode)
    state = _build_initial_state(template)
    if advanced_mode:
        _apply_advanced_overrides(st, template, state)
    else:
        st.caption("Using template defaults. Turn on Advanced mode to edit code, constraints, algorithm, or optimization settings.")
    validation = _validate_builder_state(st, state)
    _render_preview_area(st, px, state, validation)
    _render_export_area(st, advanced_mode, state, validation)


def _resolve_advanced_mode(st: Any) -> bool:
    default_advanced = bool(st.session_state.get("studio_advanced_mode", False))
    toggle_widget = getattr(st, "toggle", None)
    widget = toggle_widget if callable(toggle_widget) else st.checkbox
    advanced_mode = coerce_bool_widget_value(
        widget(
            "Advanced mode",
            value=default_advanced,
            help="Show code editors, algorithm selection, and expert settings.",
        ),
        default=default_advanced,
    )
    st.session_state["studio_advanced_mode"] = advanced_mode
    if not advanced_mode:
        st.info("Beginner mode is active. Studio will use the template defaults and the recommended NSGA-II setup.")
    return advanced_mode


def _select_template(st: Any, advanced_mode: bool) -> dict[str, str]:
    st.markdown("### Step 1. Choose a starting point")
    examples = example_objectives()
    col_cat, col_tmpl = st.columns([2, 2])
    with col_cat:
        categories = sorted({t.get("category", "other") for t in examples.values()})
        cat_labels = {"math": "Math benchmarks", "engineering": "Engineering", "ml": "Machine Learning", "scheduling": "Scheduling", "blank": "Blank"}
        selected_cat = st.selectbox(
            "Domain",
            ["all"] + categories,
            format_func=lambda c: "All templates" if c == "all" else cat_labels.get(c, c.title()),
            help="Filter templates by application domain.",
        )
    filtered = {k: v for k, v in examples.items() if selected_cat == "all" or v.get("category") == selected_cat}
    with col_tmpl:
        default_idx = list(filtered.keys()).index(_DEFAULT_TEMPLATE) if _DEFAULT_TEMPLATE in filtered else 0
        template_name = st.selectbox(
            "Start from a template",
            list(filtered.keys()),
            index=default_idx,
            help="Pick a starting point. You can edit everything below.",
        )
    template = filtered[template_name]
    description = template.get("description", "")
    difficulty = template.get("difficulty", "")
    if description:
        difficulty_badge = ""
        if difficulty:
            color = {"beginner": "green", "intermediate": "orange", "advanced": "red"}.get(difficulty, "gray")
            difficulty_badge = (
                f' <span style="background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:0.75rem;margin-left:8px">{difficulty.title()}</span>'
            )
        st.markdown(f"{description}{difficulty_badge}", unsafe_allow_html=True)
    render_template_chips(st, template, advanced_mode=advanced_mode)
    render_template_snapshot(st, template)
    return template


def _build_initial_state(template: dict[str, str]) -> dict[str, Any]:
    return {
        "algorithm": "nsgaii",
        "bounds_text": template.get("bounds", "0.0, 1.0"),
        "budget": 2000,
        "code": template["code"],
        "constraint_code": "",
        "constraint_error": None,
        "constraint_fn": None,
        "n_constraints": 0,
        "n_obj": int(template["n_obj"]),
        "n_var": int(template["n_var"]),
        "pop_size": 50,
        "problem_name": "my_problem",
        "seed": 42,
    }


def _apply_advanced_overrides(st: Any, template: dict[str, str], state: dict[str, Any]) -> None:
    with st.expander("Edit problem definition", expanded=False):
        state["problem_name"] = st.text_input("Problem name", value=state["problem_name"], help="Human-readable label for logs and exported scripts.")
        col_nvar, col_nobj = st.columns(2)
        with col_nvar:
            state["n_var"] = st.number_input("Decision variables (n_var)", min_value=1, max_value=100, value=state["n_var"], step=1, help="How many inputs does your problem have?")
        with col_nobj:
            state["n_obj"] = st.number_input("Objectives (n_obj)", min_value=2, max_value=10, value=state["n_obj"], step=1, help="How many objectives to minimize (must be >= 2).")
        state["bounds_text"] = st.text_area("Variable bounds", value=state["bounds_text"], height=68, help="One line applies to ALL variables. Or one 'lower, upper' per variable.")
        st.markdown("**Objective function** (receives `x`, return a list of objectives)")
        st.caption("Expected format: compute objective values and `return [f0, f1, ...]` with exactly `n_obj` entries.")
        state["code"] = st.text_area(
            "Objective code",
            value=state["code"],
            height=280,
            help="Write Python code that uses `x` and returns a list of n_obj values.",
            label_visibility="collapsed",
        )
        constraint_code, n_constraints, constraint_fn, constraint_error = render_constraint_section(st, template)
        state["constraint_code"] = constraint_code
        state["n_constraints"] = n_constraints
        state["constraint_fn"] = constraint_fn
        state["constraint_error"] = constraint_error

    with st.expander("Choose algorithm", expanded=False):
        st.markdown("\n".join(f"- **{v}**: {_ALGO_HELP[k]}" for k, v in _ALGO_LABELS.items()))
        state["algorithm"] = str(
            st.selectbox(
                "Algorithm",
                list(_ALGO_LABELS.keys()),
                index=0,
                format_func=lambda key: _ALGO_LABELS.get(key, key),
                help="Pick a MOEA. Start with NSGA-II unless you have a specific reason.",
            )
        )

    with st.expander("Advanced settings", expanded=False):
        col_budget, col_pop = st.columns(2)
        with col_budget:
            state["budget"] = int(st.number_input("Max evaluations", min_value=200, max_value=50000, value=state["budget"], step=500, help="Higher = better results, slower preview."))
        with col_pop:
            state["pop_size"] = int(st.number_input("Population size", min_value=10, max_value=500, value=state["pop_size"], step=10, help="Number of candidate solutions per generation."))
        state["seed"] = int(st.number_input("Seed", min_value=0, value=state["seed"], step=1, help="Random seed for reproducibility."))


def _validate_builder_state(st: Any, state: dict[str, Any]) -> dict[str, Any]:
    bounds_result = parse_bounds_text(state["bounds_text"], int(state["n_var"]))
    bounds_ok = [] if isinstance(bounds_result, str) else bounds_result
    if isinstance(bounds_result, str):
        st.error(f"Bounds error: {bounds_result}")
    compile_error: str | None = None
    fn: Any = None
    if str(state["code"]).strip():
        try:
            fn = compile_objective_function(state["code"])
        except SyntaxError as exc:
            compile_error = f"Syntax error on line {exc.lineno}: {exc.msg}"
        except Exception as exc:
            compile_error = str(exc)
    if compile_error:
        st.error(f"Code error: {compile_error}")
    return {"bounds_ok": bounds_ok, "compile_error": compile_error, "fn": fn, "has_constraint_error": state["constraint_error"] is not None}


def _render_preview_area(st: Any, px: Any, state: dict[str, Any], validation: dict[str, Any]) -> None:
    st.markdown("### Step 2. Run a quick preview")
    st.markdown(
        '<div class="studio-data-strip"><p>Preview runs use a small, fast budget so you can learn the shape of the trade-off before committing to a larger experiment.</p></div>',
        unsafe_allow_html=True,
    )
    quick_run = st.button("Run preview", type="primary", use_container_width=True)
    st.markdown("### Step 3. Read the result")
    if not quick_run:
        st.markdown(
            '<div class="studio-data-strip"><p>Your preview chart and summary table will appear here after you click <strong>Run preview</strong>.</p></div>',
            unsafe_allow_html=True,
        )
        return
    if validation["fn"] is None or not validation["bounds_ok"] or validation["compile_error"] or validation["has_constraint_error"]:
        return
    _run_and_show_preview(st, px, state=state, fn=validation["fn"], bounds_ok=validation["bounds_ok"])


def _render_export_area(st: Any, advanced_mode: bool, state: dict[str, Any], validation: dict[str, Any]) -> None:
    st.divider()
    st.subheader("Export")
    st.markdown(
        '<div class="studio-data-strip"><p>When the template or custom code is valid, Studio can turn the current setup into a standalone Python script.</p></div>',
        unsafe_allow_html=True,
    )
    if validation["fn"] is None or not validation["bounds_ok"] or validation["compile_error"] or validation["has_constraint_error"]:
        if not str(state["code"]).strip():
            st.info("Write your objective function above to enable export.")
        elif validation["compile_error"] or validation["has_constraint_error"]:
            st.warning("Fix the errors above to enable export.")
        elif not validation["bounds_ok"]:
            st.warning("Fix the bounds above to enable export.")
        else:
            st.info("Complete the fields above to enable export.")
        return

    script = generate_script(
        state["code"],
        name=state["problem_name"],
        n_var=int(state["n_var"]),
        n_obj=int(state["n_obj"]),
        bounds=validation["bounds_ok"],
        algorithm=state["algorithm"],
        budget=state["budget"],
        constraint_code=state["constraint_code"],
        n_constraints=state["n_constraints"],
    )
    st.download_button(
        "Download as Python script",
        data=script,
        file_name=f"{state['problem_name'].replace(' ', '_').lower()}.py",
        mime="text/x-python",
        help="Download a standalone .py file you can run with `python <file>.py`.",
    )
    if advanced_mode:
        with st.expander("Preview generated script"):
            st.code(script, language="python")


def _run_and_show_preview(st: Any, px: Any, *, state: dict[str, Any], fn: Any, bounds_ok: list[tuple[float, float]]) -> None:
    try:
        with st.spinner("Running optimization..."):
            preview = run_preview_optimization(
                fn,
                n_var=int(state["n_var"]),
                n_obj=int(state["n_obj"]),
                bounds=bounds_ok,
                algorithm=state["algorithm"],
                budget=state["budget"],
                pop_size=state["pop_size"],
                seed=state["seed"],
                constraints=state["constraint_fn"],
                n_constraints=state["n_constraints"],
                objective_code=state["code"],
                constraint_code=state["constraint_code"],
            )
        F = preview["F"]
        st.success(f"Found {len(F)} solutions in {preview['elapsed_ms']:.0f} ms")
        render_preview_guidance(st, int(state["n_obj"]))
        render_preview_plot(st, px, F, int(state["n_obj"]), str(state["problem_name"]))
        render_summary_table(st, F, int(state["n_obj"]))
    except TimeoutError as exc:
        st.error(str(exc))
        st.info("Try reducing max evaluations, simplifying your objective function, or removing expensive loops.")
    except Exception as exc:
        st.error(f"Optimization failed: {exc}")
        st.info("Common causes: your function returns the wrong number of objectives, or uses variables outside the bounds.")


__all__ = ["render_problem_builder"]
