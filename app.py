"""Streamlit entry point: py -m streamlit run app.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from shopsphere.app.app import main  # noqa: E402

main()
