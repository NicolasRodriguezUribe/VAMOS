"""Welcome and onboarding content for the Studio app."""

from __future__ import annotations

from typing import Any


def _render_first_launch_walkthrough(st: Any) -> None:
    if st.session_state.get("walkthrough_dismissed", False):
        return

    st.markdown(
        '<div class="walkthrough-banner">'
        "<h4>Welcome! Here's how to get started</h4>"
        "<ol>"
        "<li><b>Problem Builder</b> tab -- choose a starter template and click <b>Run preview</b>. Studio uses safe defaults for you.</li>"
        "<li>Read the chart and summary to see the trade-off between your goals.</li>"
        "<li>Need more control? Turn on <b>Advanced mode</b> later to edit code, change algorithms, or tweak settings.</li>"
        "</ol>"
        "</div>",
        unsafe_allow_html=True,
    )
    if st.button("Got it -- dismiss this guide", key="dismiss_walkthrough"):
        st.session_state["walkthrough_dismissed"] = True
        st.rerun()


def render_welcome_tab(st: Any) -> None:
    """Render the Welcome / Getting Started onboarding page."""
    _render_first_launch_walkthrough(st)
    st.header("Welcome to VAMOS Studio")
    st.markdown(
        '<div class="studio-section-intro">'
        '<span class="studio-kicker">First session</span>'
        "<h3>Start simple, then peel back the layers</h3>"
        "<p>The Studio now defaults to a guided path: start with a template, inspect a preview, then compare solutions only after you have something concrete to look at.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "VAMOS Studio helps you explore trade-offs when you have more than one goal. "
        "You can start with templates and safe defaults, then move into code only when you need it."
    )
    st.caption("Beginner flow: pick a template, run a preview, then inspect the trade-off chart.")

    col1, col2, col3 = st.columns(3)
    with col1:
        _render_step(
            st,
            step_num="1",
            title="1. Pick a starter",
            body=(
                "Open **Problem Builder**, choose a template, and stay in the default beginner mode. "
                "You do not need to write Python to get your first Pareto front."
            ),
            expander="What beginners change first",
            details=(
                "- Pick a domain template.\n"
                "- Read the plain-language description.\n"
                "- Click **Run preview**.\n"
                "- Only turn on **Advanced mode** if the template is not enough."
            ),
        )

    with col2:
        _render_step(
            st,
            step_num="2",
            title="2. Run a quick preview",
            body=(
                "Studio runs the optimizer with recommended defaults, then shows a chart and summary table. "
                "This is the fastest way to understand the shape of your trade-offs."
            ),
            expander="What the preview means",
            details=(
                "- Each point is one candidate solution.\n"
                "- Better in one goal often means worse in another.\n"
                "- The summary table helps you see the spread of results quickly."
            ),
        )

    with col3:
        _render_step(
            st,
            step_num="3",
            title="3. Compare saved runs",
            body=(
                "Use **Explore Results** to compare algorithms, rank solutions based on what matters most to you, "
                "or browse the built-in demo before you have your own results."
            ),
            expander="When to use Advanced mode",
            details=(
                "Turn it on when you want to:\n\n"
                "- edit the objective function\n"
                "- add constraints\n"
                "- change the algorithm\n"
                "- tune budget, seed, or population size"
            ),
        )

    st.divider()
    render_welcome_expanders(st)


def _render_step(st: Any, *, step_num: str, title: str, body: str, expander: str, details: str) -> None:
    st.markdown(
        f'<div class="studio-step-card"><span class="studio-step-number">{step_num}</span></div>',
        unsafe_allow_html=True,
    )
    st.subheader(title)
    st.markdown(body)
    with st.expander(expander):
        st.markdown(details)


def render_welcome_expanders(st: Any) -> None:
    """Render the collapsible quick-reference sections."""
    first_visit = not st.session_state.get("walkthrough_dismissed", False)
    with st.expander("What is multi-objective optimization?", expanded=first_visit):
        st.markdown(
            "**In short:** when you have conflicting goals (e.g. cost vs quality, "
            "speed vs accuracy), there's no single best answer. Instead, you get "
            "a set of **trade-off solutions** called the **Pareto front** -- "
            "improving one goal always means sacrificing another.\n\n"
            "VAMOS uses **evolutionary algorithms** to discover these trade-offs "
            "efficiently. The default algorithm (NSGA-II) works well for most problems. "
            "You can try others as you get more comfortable."
        )

    with st.expander("Starter tips", expanded=False):
        st.markdown(
            "- Start with **NSGA-II** unless you know you need something else.\n"
            "- Use a built-in template before writing custom code.\n"
            "- Compare only two goals at a time when reading the scatter plot.\n"
            "- Use **Advanced mode** only after the default flow feels clear."
        )

    with st.expander("For Python and CLI users", expanded=False):
        st.markdown(
            "| Command | Description |\n"
            "|---------|-------------|\n"
            "| `vamos quickstart` | Guided wizard for a first run |\n"
            "| `vamos create-problem` | Scaffold a custom problem file |\n"
            "| `vamos summarize` | Table summary of results |\n"
            "| `vamos bench` | Benchmark suite across algorithms |\n"
            "| `vamos tune` | Hyperparameter tuning |\n"
            "| `vamos check` | Verify installation and backends |\n"
            "| `vamos studio` | Launch this dashboard |\n"
            "| `vamos profile` | Performance profiling |\n"
        )
        st.markdown(
            "\n| Goal | Code |\n"
            "|------|------|\n"
            '| Run a benchmark | `optimize("zdt1", algorithm="nsgaii", max_evaluations=5000)` |\n'
            '| Custom problem | `make_problem(fn, n_var=2, n_obj=2, bounds=[(0,1),(0,1)], encoding="real")` |\n'
            '| Compare seeds | `optimize("zdt1", seed=[0, 1, 2, 3, 4])` |\n'
            "| List problems | `from vamos import available_problem_names; available_problem_names()` |\n"
        )

    with st.expander("Keyboard shortcuts (Streamlit defaults)", expanded=False):
        st.markdown(
            "| Key | Action |\n"
            "|-----|--------|\n"
            "| `R` | Rerun the app |\n"
            "| `C` | Clear cache and rerun |\n"
            "| `L` | Toggle light/dark theme |\n"
        )


__all__ = ["render_welcome_tab", "render_welcome_expanders"]
