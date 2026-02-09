"""Tests for the Studio onboarding / Welcome tab."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_streamlit():
    """Create a mock Streamlit module with chainable methods."""
    st = MagicMock()
    # columns returns a tuple of context managers
    col1, col2, col3 = MagicMock(), MagicMock(), MagicMock()
    col1.__enter__ = MagicMock(return_value=col1)
    col1.__exit__ = MagicMock(return_value=False)
    col2.__enter__ = MagicMock(return_value=col2)
    col2.__exit__ = MagicMock(return_value=False)
    col3.__enter__ = MagicMock(return_value=col3)
    col3.__exit__ = MagicMock(return_value=False)
    st.columns.return_value = (col1, col2, col3)

    exp = MagicMock()
    exp.__enter__ = MagicMock(return_value=exp)
    exp.__exit__ = MagicMock(return_value=False)
    st.expander.return_value = exp

    return st


def test_welcome_tab_renders_without_error(mock_streamlit):
    from vamos.ux.studio.app import _render_welcome_tab

    _render_welcome_tab(mock_streamlit)
    mock_streamlit.header.assert_called_once()
    assert "Welcome" in mock_streamlit.header.call_args[0][0]


def test_welcome_tab_shows_three_columns(mock_streamlit):
    from vamos.ux.studio.app import _render_welcome_tab

    _render_welcome_tab(mock_streamlit)
    mock_streamlit.columns.assert_called_once_with(3)


def test_welcome_tab_has_expanders(mock_streamlit):
    from vamos.ux.studio.app import _render_welcome_tab

    _render_welcome_tab(mock_streamlit)
    # Should create expanders for quick reference and concept explanation
    assert mock_streamlit.expander.call_count >= 3


def test_welcome_tab_mentions_cli_commands(mock_streamlit):
    from vamos.ux.studio.app import _render_welcome_tab

    _render_welcome_tab(mock_streamlit)
    # Gather all markdown calls text
    all_text = " ".join(str(c) for c in mock_streamlit.markdown.call_args_list)
    assert "quickstart" in all_text
    assert "create-problem" in all_text


def test_main_creates_three_tabs():
    """main() should now create three tabs: Welcome, Problem Builder, Explore Results."""
    from unittest.mock import patch

    from vamos.ux.studio.app import main

    st = MagicMock()
    px = MagicMock()

    # Make st.columns return the right number of context-manager mocks
    def _fake_columns(n, **kwargs):
        cols = []
        for _ in range(n if isinstance(n, int) else len(n)):
            c = MagicMock()
            c.__enter__ = MagicMock(return_value=c)
            c.__exit__ = MagicMock(return_value=False)
            cols.append(c)
        return tuple(cols)

    st.columns.side_effect = _fake_columns

    # Make expanders work as context managers
    def _fake_expander(*args, **kwargs):
        exp = MagicMock()
        exp.__enter__ = MagicMock(return_value=exp)
        exp.__exit__ = MagicMock(return_value=False)
        return exp

    st.expander.side_effect = _fake_expander

    # Make tabs return context-manager mocks
    tab1, tab2, tab3 = MagicMock(), MagicMock(), MagicMock()
    for t in (tab1, tab2, tab3):
        t.__enter__ = MagicMock(return_value=t)
        t.__exit__ = MagicMock(return_value=False)
    st.tabs.return_value = (tab1, tab2, tab3)

    import vamos.ux.studio.app as app_module

    original_st = app_module._import_streamlit
    original_px = app_module._import_plotly

    app_module._import_streamlit = lambda: st
    app_module._import_plotly = lambda: px
    try:
        with patch("vamos.ux.studio.problem_builder.render_problem_builder"):
            main(["--study-dir", "dummy"])
        # Verify tabs were created with the right labels
        st.tabs.assert_called_once()
        tab_labels = st.tabs.call_args[0][0]
        assert "Welcome" in tab_labels
        assert "Problem Builder" in tab_labels
        assert "Explore Results" in tab_labels
    finally:
        app_module._import_streamlit = original_st
        app_module._import_plotly = original_px
