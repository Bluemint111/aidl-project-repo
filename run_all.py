"""Run the complete ShopSphere pipeline from the project root."""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from shopsphere.pipelines.run_all import main  # noqa: E402

if __name__ == "__main__":
    main()
