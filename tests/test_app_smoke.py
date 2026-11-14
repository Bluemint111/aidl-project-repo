"""Smoke test for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_renders_without_exceptions():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 6
