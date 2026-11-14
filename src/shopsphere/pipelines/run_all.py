from __future__ import annotations

import os
import subprocess
import sys

from shopsphere.app import approvals
from shopsphere.common import OUT_DATA, PROJECT, ROLLING_MODELS, SRC_DIR
from shopsphere.data import validate_clean
from shopsphere.features import build_analytics, prep_contacts
from shopsphere.models import score_offers, train_offer


def _run_module(module: str, *args: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=PROJECT, env=env, check=True,
    )


def ensure_rolling_model() -> None:
    if os.environ.get("SHOPSPHERE_SKIP_ROLLING") == "1":
        print("rolling model step skipped (SHOPSPHERE_SKIP_ROLLING=1)")
        return
    meta = os.path.join(ROLLING_MODELS, "rolling_meta.json")
    live = os.path.join(OUT_DATA, "live_scores.pkl")
    retrain = os.environ.get("SHOPSPHERE_RETRAIN_ROLLING") == "1"
    if os.path.exists(meta) and os.path.exists(live) and not retrain:
        print("rolling model artifacts already present")
        return

    count = os.environ.get("SHOPSPHERE_SNAPSHOTS", "12")
    seeds = os.environ.get("SHOPSPHERE_MLP_SEEDS", "42")
    epochs = os.environ.get("SHOPSPHERE_MLP_EPOCHS", "10")
    print(f"building {count} rolling snapshots")
    _run_module("shopsphere.models.snapshot_builder", "--count", count)
    print(f"training rolling W1 model (seeds={seeds}, max epochs={epochs})")
    _run_module(
        "shopsphere.models.train_rolling",
        "--mlp-seeds", seeds, "--max-epochs", epochs,
    )


def main() -> None:
    print("=== 0/7 clean data validation ===")
    report = validate_clean.validate(strict=True)
    print(f"clean data ok: {len(report['files'])} files")
    print("=== 1/7 rolling W1 model ===")
    ensure_rolling_model()
    print("=== 2/7 contact features ===")
    prep_contacts.main()
    print("=== 3/7 offer-response model ===")
    train_offer.main()
    print("=== 4/7 candidate scoring ===")
    score_offers.main()
    print("=== 5/7 analytics ===")
    build_analytics.main()
    print("=== 6/7 approvals store ===")
    approvals.connect().close()
    print("=== 7/7 run summary ===")
    print("Pipeline complete. Run: py app.py")


if __name__ == "__main__":
    main()
