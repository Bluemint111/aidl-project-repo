"""Regression tests for the Week 8 model checkpoint deliverable.

These tests are cheap: they validate the saved artifacts and the helpers that
produce them rather than retraining the model.  Tests that need generated
artifacts skip cleanly when the pipeline has not been run yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from shopsphere.data.asof_time import black_friday, window_bounds
from shopsphere.data.build_features import days_since, to01

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "artifacts" / "features"
FIXED_REPORTS = ROOT / "artifacts" / "reports" / "fixed"
SUBMISSION = ROOT / "artifacts" / "outputs" / "submission.csv"
NOTEBOOK = ROOT / "notebooks" / "Week8_ShopSphere_Model_Checkpoint.ipynb"
CLEAN = ROOT / "data" / "clean"


def test_to01_handles_mixed_true_false_encodings():
    result = to01(pd.Series(["TRUE", "false", "Y", "no", "1", "0", None]))
    assert result.tolist()[:6] == [1, 0, 1, 0, 1, 0]
    assert pd.isna(result.iloc[6])


def test_days_since_uses_as_of():
    dates = pd.Series(["2024-09-01", "2024-09-30"])
    assert days_since(dates, pd.Timestamp("2024-09-30")).tolist() == [29, 0]


def test_window_bounds_and_black_friday():
    start, end = window_bounds(pd.Timestamp("2024-09-30"), 30)
    assert start == pd.Timestamp("2024-10-01")
    assert end == pd.Timestamp("2024-10-30")
    assert black_friday(2024) == pd.Timestamp("2024-11-29")


@pytest.mark.skipif(not (FEATURES / "train_features.pkl").exists(), reason="features not built")
def test_feature_artifacts_match_metadata():
    meta = json.loads((FEATURES / "feature_metadata.json").read_text(encoding="utf-8"))
    train = pd.read_pickle(FEATURES / "train_features.pkl")
    test = pd.read_pickle(FEATURES / "test_features.pkl")
    assert len(train) == 64000
    assert len(test) == 16000
    assert meta["n_features"] == 180
    assert len(meta["feature_columns"]) == meta["n_features"]
    assert set(meta["feature_columns"]).issubset(train.columns)
    assert set(meta["feature_columns"]).issubset(test.columns)
    assert set(train["target"].unique()) <= {0, 1}
    assert not set(train["customer_id"]) & set(test["customer_id"])


@pytest.mark.skipif(not SUBMISSION.exists(), reason="submission not generated")
def test_submission_is_valid_and_matches_metrics():
    metrics = json.loads((FIXED_REPORTS / "metrics.json").read_text(encoding="utf-8"))
    submission = pd.read_csv(SUBMISSION)
    test_ids = pd.read_csv(CLEAN / "test.csv")["customer_id"]
    assert list(submission.columns) == ["customer_id", "target"]
    assert len(submission) == len(test_ids) == 16000
    assert submission["customer_id"].is_unique
    assert set(submission["customer_id"]) == set(test_ids)
    assert set(submission["target"].unique()) <= {0, 1}
    assert int(submission["target"].sum()) == metrics["submission_positives"]


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="notebook missing")
def test_checkpoint_notebook_is_valid_and_error_free():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells
    assert all(c.get("execution_count") is not None for c in code_cells)
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not errors
