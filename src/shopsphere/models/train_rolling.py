"""Train and backtest the rolling 30-day ShopSphere propensity model.

The model uses the as-of snapshots produced by ``snapshot_builder.py``.  The
split is temporal: earlier snapshots train, a later snapshot validates, and the
most recent labelled snapshot tests.  A 30-day embargo prevents target-window
overlap between training and validation.

Outputs (artifacts/models/rolling and artifacts/reports/rolling) include:
    rolling_preprocessor.joblib, rolling_hgb.joblib, rolling_logreg.joblib,
    rolling_mlp_seed*.pt, rolling_meta.json,
    rolling_model_comparison.csv, rolling_backtest.csv, rolling_metrics.json,
    data/processed/live_scores.pkl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, log_loss,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset

from shopsphere.common import (
    ARTIFACTS_SNAPSHOTS, DATA_PROCESSED, ROLLING_MODELS, ROLLING_REPORTS,
)
from shopsphere.data import asof_features as af
from shopsphere.data.asof_time import resolve_as_of, window_bounds
from shopsphere.models.mlp_model import MLP

SNAPSHOTS = ARTIFACTS_SNAPSHOTS
MODELS = ROLLING_MODELS
REPORTS = ROLLING_REPORTS
LIVE_FEATURES = os.path.join(DATA_PROCESSED, "live_features.pkl")
LIVE_SCORES = os.path.join(DATA_PROCESSED, "live_scores.pkl")

SEED = 42
CATEGORICAL = ["cust_country", "cust_region", "cust_tier", "geo_tier",
               "pref_channel", "pref_language"]
DROP_COLUMNS = {"customer_id", "target", "as_of", "_as_of",
                "target_window_start", "target_window_end", "last_contact_date"}


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_snapshots(snapshot_dir: str) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    paths = sorted(glob.glob(os.path.join(snapshot_dir, "snapshot_*.pkl")))
    if not paths:
        raise FileNotFoundError(f"No snapshots in {snapshot_dir}. Run snapshot_builder.py first.")
    out = []
    for path in paths:
        df = pd.read_pickle(path)
        if "target" not in df.columns:
            raise ValueError(f"{path} has no target column")
        out.append((pd.Timestamp(df["as_of"].iloc[0]).normalize(), df))
    out.sort(key=lambda item: item[0])
    return out


def feature_columns(frames: list[pd.DataFrame]) -> tuple[list[str], list[str], list[str]]:
    cols = [c for c in frames[0].columns if c not in DROP_COLUMNS]
    for frame in frames[1:]:
        cols = [c for c in cols if c in frame.columns]
    cats = [c for c in CATEGORICAL if c in cols]
    nums = [c for c in cols if c not in cats]
    return cols, nums, cats


def make_preprocessor(nums: list[str], cats: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=50, sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, nums),
        ("cat", cat_pipe, cats),
    ])


def xy(df: pd.DataFrame, features: list[str], cats: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    X = df[features].copy()
    for col in cats:
        X[col] = X[col].astype("string")
    for col in features:
        if col not in cats:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X, df["target"].to_numpy(dtype=int)


def best_f1(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = (2 * precision * recall) / np.where(precision + recall == 0, np.nan, precision + recall)
    idx = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return (float(thresholds[idx]) if len(thresholds) else 0.5), float(f1[idx])


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    thr, f1 = best_f1(y, p)
    pred = (p >= thr).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 5),
        "pr_auc": round(float(average_precision_score(y, p)), 5),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 5),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 5),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 5),
        "threshold": round(float(thr), 5),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "log_loss": round(float(log_loss(y, p)), 5),
        "n": int(len(y)),
        "positive_rate": round(float(np.mean(y)), 5),
    }


def train_mlp(X_tr, y_tr, X_va, y_va, seed: int, max_epochs: int = 30,
              batch_size: int = 512, patience: int = 5):
    torch.manual_seed(seed)
    model = MLP(X_tr.shape[1])
    pos_weight = torch.tensor(
        [(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)], dtype=torch.float32
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=2
    )
    ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X_tr)),
        torch.from_numpy(y_tr).float(),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    Xva_t = torch.from_numpy(np.ascontiguousarray(X_va))

    best_ap, best_state, best_epoch, wait = -1.0, None, 0, 0
    log = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(Xva_t)).numpy()
        val_ap = average_precision_score(y_va, val_prob)
        val_auc = roc_auc_score(y_va, val_prob)
        sched.step(val_ap)
        log.append({
            "seed": seed, "epoch": epoch,
            "loss": round(total / len(ds), 5),
            "val_ap": round(float(val_ap), 5),
            "val_auc": round(float(val_auc), 5),
        })
        print(f"  [seed {seed}] epoch {epoch:02d} loss={total/len(ds):.4f} "
              f"val_ap={val_ap:.4f} val_auc={val_auc:.4f}")
        if val_ap > best_ap + 1e-5:
            best_ap, best_epoch, wait = val_ap, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"  [seed {seed}] early stop at {epoch} (best {best_epoch})")
                break
    model.load_state_dict(best_state)
    return model, best_epoch, log


def predict_mlp(model, X, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch]))
            out.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(out)


def fit_full_mlp(X, y, seed: int, epochs: int, batch_size: int = 512):
    torch.manual_seed(seed)
    model = MLP(X.shape[1])
    pos_weight = torch.tensor(
        [(len(y) - y.sum()) / max(y.sum(), 1)], dtype=torch.float32
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X)),
        torch.from_numpy(y).float(),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for epoch in range(1, max(1, int(epochs)) + 1):
        model.train()
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        if epoch % 5 == 0 or epoch == int(epochs):
            print(f"  [full seed {seed}] epoch {epoch:02d} loss={total/len(ds):.4f}")
    return model


def cap_rows(df: pd.DataFrame, max_rows: int | None, seed: int = SEED) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    pos = df[df["target"] == 1]
    neg = df[df["target"] == 0]
    n_pos = min(len(pos), int(max_rows * len(pos) / len(df)) + 1)
    n_neg = max_rows - n_pos
    pos = resample(pos, replace=False, n_samples=n_pos, random_state=seed)
    neg = resample(neg, replace=False, n_samples=min(n_neg, len(neg)), random_state=seed)
    return pd.concat([pos, neg], ignore_index=True)


def select_split(snapshots, val_n: int = 1, test_n: int = 1, embargo_days: int = 30):
    if len(snapshots) < 3:
        raise ValueError("Need at least three snapshots for train/validation/test")
    test = snapshots[-test_n:]
    val = snapshots[-(val_n + test_n):-test_n]
    val_start = val[0][0]
    train = [
        item for item in snapshots[:-(val_n + test_n)]
        if item[0] + pd.Timedelta(days=embargo_days) < val_start
    ]
    if not train:
        train = snapshots[:-(val_n + test_n)]
    return train, val, test


def concat(frames) -> pd.DataFrame:
    return pd.concat([frame for _, frame in frames], ignore_index=True)


def model_probabilities(models, pre, X):
    mlp = np.mean([predict_mlp(m, X) for m in models["mlp"]], axis=0) if models["mlp"] else None
    hgb = models["hgb"].predict_proba(X)[:, 1]
    logreg = models["logreg"].predict_proba(X)[:, 1]
    out = {
        "Logistic Regression (baseline)": logreg,
        "Gradient Boosting (HGB)": hgb,
    }
    if mlp is not None:
        out["Deep Learning MLP (seed ensemble)"] = mlp
        out["Ensemble (MLP + HGB)"] = (mlp + hgb) / 2
        out["Ensemble (MLP + HGB + LogReg)"] = (mlp + hgb + logreg) / 3
    return out


def score_live_only(args) -> dict:
    """Score an existing live snapshot with the saved rolling model."""
    from shopsphere.models.scoring import RollingScorer

    scorer = RollingScorer(args.out_dir)
    if os.path.exists(args.live_features):
        live = pd.read_pickle(args.live_features)
    else:
        raw = af.load_raw()
        latest = resolve_as_of(None, af.DATA)
        live = af.build_snapshot(raw, latest, af.DATA, labelled=False)
    live = live.copy()
    live["w1_probability"] = scorer.predict_proba(live)
    live["w1_target"] = (live["w1_probability"] >= scorer.threshold).astype(int)
    if "_as_of" in live.columns and pd.notna(live["_as_of"].iloc[0]):
        as_of = pd.Timestamp(live["_as_of"].iloc[0]).normalize()
    else:
        as_of = resolve_as_of(None, af.DATA)
    start, end = window_bounds(as_of, af.WINDOW_DAYS)
    live["w1_as_of"] = as_of
    live["w1_window_start"] = start
    live["w1_window_end"] = end
    out_path = getattr(args, "live_out", None) or LIVE_SCORES
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    live.to_pickle(out_path)
    report = {
        "scored_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model": scorer.selected,
        "threshold": scorer.threshold,
        "as_of": str(as_of.date()),
        "window_start": str(start.date()),
        "window_end": str(end.date()),
        "rows": int(len(live)),
        "mean_probability": round(float(live["w1_probability"].mean()), 5),
        "high_propensity_share": round(float(live["w1_target"].mean()), 5),
        "path": out_path,
    }
    os.makedirs(args.reports_dir, exist_ok=True)
    with open(os.path.join(args.reports_dir, "live_scoring.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rolling ShopSphere W1 model")
    parser.add_argument("--snapshots-dir", default=SNAPSHOTS)
    parser.add_argument("--live-features", default=LIVE_FEATURES)
    parser.add_argument("--live-out", default=LIVE_SCORES)
    parser.add_argument("--out-dir", default=MODELS)
    parser.add_argument("--reports-dir", default=REPORTS)
    parser.add_argument("--live-only", action="store_true",
                        help="Score the live snapshot with existing models and exit")
    parser.add_argument("--val-snapshots", type=int, default=1)
    parser.add_argument("--test-snapshots", type=int, default=1)
    parser.add_argument("--embargo-days", type=int, default=30)
    parser.add_argument("--mlp-seeds", default="42",
                        help="Comma-separated seeds; use 42,7,2024 for a seed ensemble")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--max-train-rows", type=int, default=250000)
    parser.add_argument("--max-final-rows", type=int, default=250000)
    parser.add_argument("--max-final-epochs", type=int, default=8)
    parser.add_argument("--skip-importance", action="store_true")
    parser.add_argument("--skip-live", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)
    if args.live_only:
        print(json.dumps(score_live_only(args), indent=2))
        return
    seeds = [int(s) for s in str(args.mlp_seeds).split(",") if s.strip()]
    set_seeds(SEED)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    snapshots = load_snapshots(args.snapshots_dir)
    train_s, val_s, test_s = select_split(
        snapshots, args.val_snapshots, args.test_snapshots, args.embargo_days
    )
    print("train snapshots:", [str(d.date()) for d, _ in train_s])
    print("val snapshots:  ", [str(d.date()) for d, _ in val_s])
    print("test snapshots: ", [str(d.date()) for d, _ in test_s])

    features, nums, cats = feature_columns([df for _, df in snapshots])
    train = cap_rows(concat(train_s), args.max_train_rows)
    val = concat(val_s)
    test = concat(test_s)
    X_tr, y_tr = xy(train, features, cats)
    X_va, y_va = xy(val, features, cats)
    X_te, y_te = xy(test, features, cats)
    print(f"rows train={len(train):,} val={len(val):,} test={len(test):,} "
          f"features={len(features)}")

    pre = make_preprocessor(nums, cats)
    Xtr = pre.fit_transform(X_tr).astype(np.float32)
    Xva = pre.transform(X_va).astype(np.float32)
    Xte = pre.transform(X_te).astype(np.float32)
    print("transformed width:", Xtr.shape[1])

    models = {"mlp": []}
    models["dummy"] = DummyClassifier(strategy="prior").fit(Xtr, y_tr)
    models["logreg"] = LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=SEED
    ).fit(Xtr, y_tr)
    models["hgb"] = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, class_weight="balanced", random_state=SEED,
    ).fit(Xtr, y_tr)

    epochs = {}
    training_log = []
    if seeds:
        print("training rolling MLP ...")
        for seed in seeds:
            model, best_epoch, log = train_mlp(
                Xtr, y_tr, Xva, y_va, seed,
                max_epochs=args.max_epochs, patience=5,
            )
            models["mlp"].append(model)
            epochs[seed] = best_epoch
            training_log.extend(log)

    val_probs = model_probabilities(models, pre, Xva)
    model_results = []
    for name, p in val_probs.items():
        result = {"model": name, **metrics(y_va, p)}
        model_results.append(result)
        print(f"{name:36} PR-AUC={result['pr_auc']:.4f} ROC-AUC={result['roc_auc']:.4f}")

    candidate = {r["model"]: r["pr_auc"] for r in model_results
                 if r["model"] != "Dummy (prior)"}
    selected = max(candidate, key=candidate.get)
    selected_result = next(r for r in model_results if r["model"] == selected)
    threshold = float(selected_result["threshold"])
    print(f"selected: {selected} threshold={threshold:.4f}")

    # Backtest selected model on each validation/test snapshot.
    backtest = []
    for label, frames in (("validation", val_s), ("test", test_s)):
        for as_of, frame in frames:
            Xf, yf = xy(frame, features, cats)
            probs = model_probabilities(models, pre, pre.transform(Xf).astype(np.float32))
            p = probs[selected]
            row = {"split": label, "as_of": str(as_of.date()), **metrics(yf, p)}
            backtest.append(row)
            print(f"backtest {label} {as_of.date()} PR-AUC={row['pr_auc']:.4f}")

    # Refit final models on all labelled snapshots (metrics remain out-of-time).
    all_labelled = cap_rows(concat(snapshots), args.max_final_rows, seed=SEED + 1)
    X_all, y_all = xy(all_labelled, features, cats)
    pre_final = make_preprocessor(nums, cats)
    X_all_t = pre_final.fit_transform(X_all).astype(np.float32)
    joblib.dump(pre_final, os.path.join(args.out_dir, "rolling_preprocessor.joblib"))

    logreg_final = LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=SEED
    ).fit(X_all_t, y_all)
    joblib.dump(logreg_final, os.path.join(args.out_dir, "rolling_logreg.joblib"))

    hgb_final = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, class_weight="balanced", random_state=SEED,
    ).fit(X_all_t, y_all)
    joblib.dump(hgb_final, os.path.join(args.out_dir, "rolling_hgb.joblib"))

    final_models = {"mlp": [], "hgb": hgb_final, "logreg": logreg_final}
    for seed in seeds:
        full_epochs = min(args.max_final_epochs, max(3, int(epochs.get(seed, 6))))
        model = fit_full_mlp(X_all_t, y_all, seed, epochs=full_epochs)
        torch.save(model.state_dict(), os.path.join(args.out_dir, f"rolling_mlp_seed{seed}.pt"))
        final_models["mlp"].append(model)

    meta = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "window_days": af.WINDOW_DAYS,
        "train_snapshots": [str(d.date()) for d, _ in train_s],
        "validation_snapshots": [str(d.date()) for d, _ in val_s],
        "test_snapshots": [str(d.date()) for d, _ in test_s],
        "embargo_days": args.embargo_days,
        "rows": {
            "train": int(len(train)), "validation": int(len(val)),
            "test": int(len(test)), "all_labelled": int(len(all_labelled)),
        },
        "features": features,
        "numeric": nums,
        "categorical": cats,
        "selected_model": selected,
        "selected_threshold": threshold,
        "mlp_seeds": seeds,
        "mlp_best_epochs": {str(k): int(v) for k, v in epochs.items()},
        "transformed_width": int(Xtr.shape[1]),
        "final_transformed_width": int(X_all_t.shape[1]),
        "model_results": model_results,
    }
    with open(os.path.join(args.out_dir, "rolling_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    pd.DataFrame(model_results).to_csv(
        os.path.join(args.reports_dir, "rolling_model_comparison.csv"), index=False
    )
    pd.DataFrame(backtest).to_csv(
        os.path.join(args.reports_dir, "rolling_backtest.csv"), index=False
    )
    if training_log:
        pd.DataFrame(training_log).to_csv(
            os.path.join(args.reports_dir, "rolling_mlp_training_log.csv"), index=False
        )

    # Permutation importance for the final HGB (sample to keep it quick).
    if not args.skip_importance:
        sample_raw = X_va[: min(5000, len(Xva))]
        sample = pre_final.transform(sample_raw).astype(np.float32)
        y_sample = y_va[: len(sample)]
        imp = permutation_importance(
            hgb_final, sample, y_sample, scoring="average_precision",
            n_repeats=2, random_state=SEED, n_jobs=-1,
        )
        names = pre_final.get_feature_names_out()
        pd.DataFrame({"feature": names, "importance": imp.importances_mean}) \
            .sort_values("importance", ascending=False).head(50) \
            .to_csv(os.path.join(args.reports_dir, "rolling_feature_importance.csv"), index=False)

    # Score the live snapshot with the refit models.
    if not args.skip_live:
        live_path = args.live_features
        if os.path.exists(live_path):
            live = pd.read_pickle(live_path)
        else:
            raw = af.load_raw()
            latest = resolve_as_of(None, af.DATA)
            live = af.build_snapshot(raw, latest, af.DATA, labelled=False)
        X_live, _ = xy(live.assign(target=0), features, cats)
        X_live_t = pre_final.transform(X_live).astype(np.float32)
        probs = model_probabilities(final_models, pre_final, X_live_t)
        p = probs[selected]
        live = live.copy()
        live["w1_probability"] = p
        live["w1_target"] = (p >= threshold).astype(int)
        live["w1_as_of"] = pd.Timestamp(live["_as_of"].iloc[0]) if "_as_of" in live else pd.NaT
        start, end = window_bounds(live["w1_as_of"].iloc[0], af.WINDOW_DAYS)
        live["w1_window_start"] = start
        live["w1_window_end"] = end
        live.to_pickle(LIVE_SCORES)
        print(f"live scores -> {LIVE_SCORES} (rows={len(live):,})")

    summary = {
        "selected_model": selected,
        "selected_threshold": threshold,
        "validation": selected_result,
        "backtest": backtest,
        "train_snapshots": meta["train_snapshots"],
        "validation_snapshots": meta["validation_snapshots"],
        "test_snapshots": meta["test_snapshots"],
        "rows": meta["rows"],
    }
    with open(os.path.join(args.reports_dir, "rolling_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
