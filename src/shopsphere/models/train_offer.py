from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, log_loss,
    precision_recall_curve, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shopsphere.common import (
    CATEGORICAL_FEATURES, MODEL_FEATURES, OFFER_MODELS as OUT_MODELS,
    OFFER_REPORTS as OUT_REPORTS, OUT_DATA, ensure_dirs, read_csv, sample_ref,
)
from shopsphere.data.asof_time import parse_datetime

SEED = 42


def build_preprocessor() -> ColumnTransformer:
    numeric = [c for c in MODEL_FEATURES if c not in CATEGORICAL_FEATURES]
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ])


def best_f1(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = (2 * precision * recall) / np.where(precision + recall == 0, np.nan, precision + recall)
    idx = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    thr = float(thresholds[idx]) if len(thresholds) else 0.5
    return thr, float(f1[idx])


def score_metrics(y, p) -> dict:
    thr, f1 = best_f1(y, p)
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 5),
        "pr_auc": round(float(average_precision_score(y, p)), 5),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "log_loss": round(float(log_loss(y, p)), 5),
        "best_f1": round(f1, 5),
        "best_threshold": round(thr, 5),
        "positive_rate": round(float(np.mean(y)), 5),
        "n": int(len(y)),
    }


def make_purchase_models():
    return {
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=30, class_weight="balanced", random_state=SEED,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=SEED,
        ),
    }


def make_redemption_model():
    return HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, class_weight="balanced", random_state=SEED,
    )


def per_offer_table(df: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> list[dict]:
    out = []
    for offer, idx in df.groupby("offer_code").groups.items():
        pos = df.index.get_indexer(idx)
        yy = y[pos]
        pp = p[pos]
        row = {
            "offer_code": offer,
            "n": int(len(yy)),
            "observed_purchase_rate": round(float(yy.mean()), 5),
            "mean_prediction": round(float(pp.mean()), 5),
        }
        if len(np.unique(yy)) > 1:
            row["roc_auc"] = round(float(roc_auc_score(yy, pp)), 5)
            row["pr_auc"] = round(float(average_precision_score(yy, pp)), 5)
        else:
            row["roc_auc"] = None
            row["pr_auc"] = None
        out.append(row)
    return out


def main() -> None:
    ensure_dirs()
    df = pd.read_pickle(os.path.join(OUT_DATA, "historical_contacts.pkl"))
    df["contact_date"] = parse_datetime(df["contact_date"])
    y_purchase = df["purchased_within_14d"].to_numpy(dtype=int)

    split_value = os.environ.get("SHOPSPHERE_OFFER_SPLIT")
    if split_value:
        split_date = pd.Timestamp(split_value)
    else:
        valid_days = int(os.environ.get("SHOPSPHERE_OFFER_VALID_DAYS", "30"))
        split_date = df["contact_date"].max().normalize() - pd.Timedelta(days=valid_days)
    print(f"offer train/valid split: {split_date.date()}")
    train_mask = df["contact_date"] <= split_date
    valid_mask = ~train_mask
    train = df.loc[train_mask].copy()
    valid = df.loc[valid_mask].copy()
    y_tr = y_purchase[train_mask.to_numpy()]
    y_va = y_purchase[valid_mask.to_numpy()]
    print(f"train={len(train):,} ({y_tr.mean():.3%}) valid={len(valid):,} ({y_va.mean():.3%})")

    pre = build_preprocessor()
    Xtr = pre.fit_transform(train[MODEL_FEATURES]).astype(np.float32)
    Xva = pre.transform(valid[MODEL_FEATURES]).astype(np.float32)
    print("transformed width", Xtr.shape[1])

    results = {}
    valid_prob = {}
    fitted = {}
    for name, model in make_purchase_models().items():
        print("Training", name)
        model.fit(Xtr, y_tr)
        p = model.predict_proba(Xva)[:, 1]
        results[name] = score_metrics(y_va, p)
        valid_prob[name] = p
        fitted[name] = model
        print(f"  {name}: ROC-AUC={results[name]['roc_auc']:.4f} "
              f"PR-AUC={results[name]['pr_auc']:.4f} F1={results[name]['best_f1']:.4f}")

    selected = max(results, key=lambda n: results[n]["pr_auc"])
    threshold = results[selected]["best_threshold"]
    print(f"Selected purchase model: {selected} PR-AUC={results[selected]['pr_auc']:.4f}")

    y_offer = valid["offer_code"].to_numpy()
    per_offer = per_offer_table(valid.reset_index(drop=True), y_va, valid_prob[selected])

    red_mask = (valid["offer_code"] != "OFF-NONE") & valid["redeemed"].notna()
    red = valid.loc[red_mask]
    y_red = red["redeemed"].to_numpy(dtype=int)
    Xred = pre.transform(red[MODEL_FEATURES]).astype(np.float32)
    red_model = make_redemption_model()
    red_model.fit(Xred, y_red)
    p_red = red_model.predict_proba(Xred)[:, 1]
    red_metrics = score_metrics(y_red, p_red)
    print(f"Redemption model: n={len(red):,} ROC-AUC={red_metrics['roc_auc']:.4f} "
          f"PR-AUC={red_metrics['pr_auc']:.4f}")

    full_pre = build_preprocessor()
    Xfull = full_pre.fit_transform(df[MODEL_FEATURES]).astype(np.float32)
    final_models = {}
    for name, model in make_purchase_models().items():
        final = model.__class__(**model.get_params())
        final.fit(Xfull, y_purchase)
        final_models[name] = final
        joblib.dump(final, os.path.join(OUT_MODELS, f"purchase_{name.lower().replace(' ', '_')}.joblib"))
    joblib.dump(final_models[selected], os.path.join(OUT_MODELS, "purchase_model.joblib"))
    joblib.dump(full_pre, os.path.join(OUT_MODELS, "offer_preprocessor.joblib"))

    final_red_mask = (df["offer_code"] != "OFF-NONE") & df["redeemed"].notna()
    red_full = make_redemption_model()
    red_full.fit(Xfull[final_red_mask.to_numpy()],
                 df.loc[final_red_mask, "redeemed"].to_numpy(dtype=int))
    joblib.dump(red_full, os.path.join(OUT_MODELS, "redemption_model.joblib"))

    val_out = valid[["customer_id", "contact_date", "offer_code", "channel",
                     "cust_tier", "cust_country", "purchased_within_14d"]].copy()
    val_out["sample_ref"] = sample_ref(val_out["customer_id"])
    val_out = val_out.drop(columns=["customer_id"])
    val_out["purchase_probability"] = valid_prob[selected]
    val_out["selected_model"] = selected
    val_out.to_csv(os.path.join(OUT_REPORTS, "offer_validation_predictions.csv"), index=False)
    pd.DataFrame(per_offer).to_csv(os.path.join(OUT_REPORTS, "offer_validation_by_offer.csv"),
                                   index=False)

    metrics = {
        "seed": SEED,
        "split": {
            "train": int(len(train)), "validation": int(len(valid)),
            "train_positive_rate": round(float(y_tr.mean()), 5),
            "validation_positive_rate": round(float(y_va.mean()), 5),
        },
        "purchase_models": results,
        "selected_purchase_model": selected,
        "selected_threshold": threshold,
        "purchase_by_offer": per_offer,
        "redemption": red_metrics,
        "n_features": len(MODEL_FEATURES),
        "n_transformed": int(Xtr.shape[1]),
    }
    with open(os.path.join(OUT_REPORTS, "offer_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    meta = {
        "feature_columns": MODEL_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "numeric": [c for c in MODEL_FEATURES if c not in CATEGORICAL_FEATURES],
        "selected_purchase_model": selected,
        "threshold": threshold,
        "trained_through": str(df["contact_date"].max().date()),
    }
    with open(os.path.join(OUT_MODELS, "offer_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(metrics, indent=2)[:4000])


if __name__ == "__main__":
    main()
