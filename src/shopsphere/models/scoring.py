"""Load the trained ShopSphere propensity model and score customer feature rows.

Used by the dashboard and (later) the agent so there is a single scoring API.
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from shopsphere.common import (
    ARTIFACTS_FEATURES as FEAT, FIXED_MODELS as MODELS, FIXED_REPORTS as REPORTS,
    ROLLING_MODELS,
)
from shopsphere.models.mlp_model import MLP


class Scorer:
    def __init__(self) -> None:
        with open(os.path.join(FEAT, "feature_metadata.json"), encoding="utf-8") as fh:
            self.meta = json.load(fh)
        with open(os.path.join(REPORTS, "metrics.json"), encoding="utf-8") as fh:
            self.metrics = json.load(fh)
        with open(os.path.join(MODELS, "mlp_meta.json"), encoding="utf-8") as fh:
            self.mlp_meta = json.load(fh)
        self.features = self.meta["feature_columns"]
        self.selected = self.metrics["selected_model"]
        self.threshold = float(self.metrics["selected_threshold"])
        self.pre = joblib.load(os.path.join(MODELS, "preprocessor.joblib"))
        self.hgb = joblib.load(os.path.join(MODELS, "hgb.joblib"))
        self.logreg = joblib.load(os.path.join(MODELS, "logreg.joblib"))
        self.mlps = []
        for seed in self.mlp_meta["seeds"]:
            m = MLP(self.mlp_meta["n_in"])
            m.load_state_dict(torch.load(os.path.join(MODELS, f"mlp_seed{seed}.pt"),
                                         map_location="cpu"))
            m.eval()
            self.mlps.append(m)

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.pre.transform(df[self.features]).astype("float32")

    def _mlp_probs(self, X: np.ndarray) -> np.ndarray:
        probs = []
        with torch.no_grad():
            for m in self.mlps:
                probs.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(X)))).numpy())
        return np.mean(probs, axis=0)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._transform(df)
        mlp = self._mlp_probs(X)
        hgb = self.hgb.predict_proba(X)[:, 1]
        logreg = self.logreg.predict_proba(X)[:, 1]
        table = {
            "Logistic Regression (baseline)": logreg,
            "Gradient Boosting (baseline)": hgb,
            "Deep Learning MLP (seed ensemble)": mlp,
            "Ensemble (MLP + HGB)": (mlp + hgb) / 2,
            "Ensemble (MLP + HGB + LogReg)": (mlp + hgb + logreg) / 3,
        }
        return table[self.selected]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= self.threshold).astype(int)


class RollingScorer:
    """Scorer for the as-of/rolling W1 model in artifacts/models/rolling."""

    def __init__(self, models_dir: str | None = None) -> None:
        self.models = models_dir or ROLLING_MODELS
        with open(os.path.join(self.models, "rolling_meta.json"), encoding="utf-8") as fh:
            self.meta = json.load(fh)
        self.features = self.meta["features"]
        self.selected = self.meta["selected_model"]
        self.threshold = float(self.meta["selected_threshold"])
        self.pre = joblib.load(os.path.join(self.models, "rolling_preprocessor.joblib"))
        self.hgb = joblib.load(os.path.join(self.models, "rolling_hgb.joblib"))
        self.logreg = joblib.load(os.path.join(self.models, "rolling_logreg.joblib"))
        width = self.meta.get("final_transformed_width", self.meta["transformed_width"])
        self.mlps = []
        for seed in self.meta.get("mlp_seeds", []):
            path = os.path.join(self.models, f"rolling_mlp_seed{seed}.pt")
            if not os.path.exists(path):
                continue
            model = MLP(width)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            self.mlps.append(model)

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].copy()
        for col in self.meta["categorical"]:
            X[col] = X[col].astype("string")
        for col in self.meta["numeric"]:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        return self.pre.transform(X).astype("float32")

    def _mlp_probs(self, X: np.ndarray) -> np.ndarray:
        if not self.mlps:
            return np.full(len(X), np.nan)
        probs = []
        with torch.no_grad():
            for model in self.mlps:
                probs.append(
                    torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(X)))).numpy()
                )
        return np.mean(probs, axis=0)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._transform(df)
        mlp = self._mlp_probs(X)
        hgb = self.hgb.predict_proba(X)[:, 1]
        logreg = self.logreg.predict_proba(X)[:, 1]
        table = {
            "Logistic Regression (baseline)": logreg,
            "Gradient Boosting (HGB)": hgb,
            "Deep Learning MLP (seed ensemble)": mlp,
            "Ensemble (MLP + HGB)": (mlp + hgb) / 2,
            "Ensemble (MLP + HGB + LogReg)": (mlp + hgb + logreg) / 3,
        }
        return table[self.selected]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= self.threshold).astype(int)
