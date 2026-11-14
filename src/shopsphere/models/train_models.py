"""
ShopSphere - train baselines, a seed-ensembled PyTorch MLP and blended
ensembles for the 30-day purchase target. Writes metrics, per-model
validation/test probabilities, models and submission.csv.

Usage: py -m shopsphere.models.train_models (with src on PYTHONPATH)
"""

from __future__ import annotations

import json
import os
import random

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
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from shopsphere.common import (
    ARTIFACTS_FEATURES as FEAT, FIXED_MODELS as MODELS,
    FIXED_REPORTS as REPORTS, OUT_OUTPUTS,
)
from shopsphere.models.mlp_model import MLP

SEED = 42
MLP_SEEDS = [42, 7, 2024]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load():
    train = pd.read_pickle(os.path.join(FEAT, "train_features.pkl"))
    test = pd.read_pickle(os.path.join(FEAT, "test_features.pkl"))
    with open(os.path.join(FEAT, "feature_metadata.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    return train, test, meta


def build_preprocessor(meta: dict) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100,
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, meta["numeric"]),
        ("cat", cat_pipe, meta["categorical"]),
    ])


def metrics_at(y, prob, thr):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(thr), 4),
        "roc_auc": round(float(roc_auc_score(y, prob)), 5),
        "pr_auc": round(float(average_precision_score(y, prob)), 5),
        "log_loss": round(float(log_loss(y, prob)), 5),
        "brier": round(float(brier_score_loss(y, prob)), 5),
        "accuracy": round(float(accuracy_score(y, pred)), 5),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 5),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 5),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 5),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 5),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def evaluate(name, y, prob):
    p, r, t = precision_recall_curve(y, prob)
    f1s = (2 * p * r) / np.where((p + r) == 0, np.nan, p + r)
    best_idx = int(np.nanargmax(f1s[:-1])) if len(t) else 0
    best_thr = float(t[best_idx]) if len(t) else 0.5
    out = {"model": name, "at_0.5": metrics_at(y, prob, 0.5),
           "at_best_f1": metrics_at(y, prob, best_thr)}
    a = out["at_best_f1"]
    print(f"{name:36} ROC-AUC={a['roc_auc']:.4f} PR-AUC={a['pr_auc']:.4f} "
          f"F1={a['f1']:.4f} @thr={best_thr:.3f}")
    return out


def train_mlp(X_tr, y_tr, X_va, y_va, seed, max_epochs=80, batch_size=512, patience=10):
    torch.manual_seed(seed)
    model = MLP(X_tr.shape[1])
    pos_weight = torch.tensor([(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)

    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(X_tr)), torch.from_numpy(y_tr).float())
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
        log.append({"seed": seed, "epoch": epoch, "loss": round(total / len(ds), 5),
                    "val_ap": round(float(val_ap), 5), "val_auc": round(float(val_auc), 5)})
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
    return model, best_epoch, float(best_ap), predict_mlp(model, X_va), log


def predict_mlp(model, X, batch=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch]))
            out.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(out)


def fit_full_mlp(X, y, seed, epochs):
    torch.manual_seed(seed)
    model = MLP(X.shape[1])
    pos_weight = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(X)), torch.from_numpy(y).float())
    dl = DataLoader(ds, batch_size=512, shuffle=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        if epoch % 10 == 0 or epoch == epochs:
            print(f"  [full seed {seed}] epoch {epoch:02d} loss={total/len(ds):.4f}")
    return model


def make_hgb():
    return HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        class_weight="balanced", random_state=SEED,
    )


def main():
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(REPORTS, exist_ok=True)
    train, test, meta = load()
    feats = meta["feature_columns"]

    X = train[feats]
    y = train["target"].to_numpy()
    idx_tr, idx_va = train_test_split(
        np.arange(len(train)), test_size=0.2, stratify=y, random_state=SEED
    )
    X_tr, X_va = X.iloc[idx_tr], X.iloc[idx_va]
    y_tr, y_va = y[idx_tr], y[idx_va]
    val_ids = train["customer_id"].iloc[idx_va].to_numpy()
    X_test = test[feats]
    print(f"train={len(X_tr):,} val={len(X_va):,} test={len(X_test):,} pos_rate={y.mean():.4f}")

    pre = build_preprocessor(meta)
    Xtr = pre.fit_transform(X_tr).astype(np.float32)
    Xva = pre.transform(X_va).astype(np.float32)
    Xte = pre.transform(X_test).astype(np.float32)
    print(f"transformed width = {Xtr.shape[1]}")

    probs = {}
    results = []

    dummy = DummyClassifier(strategy="prior").fit(Xtr, y_tr)
    probs["Dummy (prior)"] = dummy.predict_proba(Xva)[:, 1]

    logreg = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, y_tr)
    probs["Logistic Regression (baseline)"] = logreg.predict_proba(Xva)[:, 1]

    hgb = make_hgb().fit(Xtr, y_tr)
    probs["Gradient Boosting (baseline)"] = hgb.predict_proba(Xva)[:, 1]

    print("\nTraining MLP seed ensemble ...")
    mlp_models, mlp_epochs, mlp_log = {}, {}, []
    mlp_val = []
    for seed in MLP_SEEDS:
        m, be, ap, vp, log = train_mlp(Xtr, y_tr, Xva, y_va, seed)
        mlp_models[seed], mlp_epochs[seed] = m, be
        mlp_val.append(vp)
        mlp_log += log
    probs["Deep Learning MLP (seed ensemble)"] = np.mean(mlp_val, axis=0)

    probs["Ensemble (MLP + HGB)"] = (probs["Deep Learning MLP (seed ensemble)"]
                                     + probs["Gradient Boosting (baseline)"]) / 2
    probs["Ensemble (MLP + HGB + LogReg)"] = (
        probs["Deep Learning MLP (seed ensemble)"]
        + probs["Gradient Boosting (baseline)"]
        + probs["Logistic Regression (baseline)"]
    ) / 3

    for name, p in probs.items():
        results.append(evaluate(name, y_va, p))

    candidate = {n: average_precision_score(y_va, p) for n, p in probs.items()
                 if not n.startswith("Dummy")}
    best_name = max(candidate, key=candidate.get)
    best_result = next(r for r in results if r["model"] == best_name)
    best_thr = best_result["at_best_f1"]["threshold"]
    print(f"\nSelected: {best_name}  val PR-AUC={candidate[best_name]:.4f}  threshold={best_thr:.3f}")

    print("\nPermutation importance (HGB) ...")
    rng = np.random.RandomState(SEED)
    sel = rng.choice(len(Xva), size=min(8000, len(Xva)), replace=False)
    imp = permutation_importance(hgb, Xva[sel], y_va[sel], scoring="average_precision",
                                 n_repeats=3, random_state=SEED, n_jobs=-1)
    names = pre.get_feature_names_out()
    pd.DataFrame({"feature": names, "importance": imp.importances_mean}) \
        .sort_values("importance", ascending=False).head(40) \
        .to_csv(os.path.join(REPORTS, "feature_importance_hgb.csv"), index=False)

    print("\nRefitting on full training set ...")
    pre_full = build_preprocessor(meta)
    Xfull = pre_full.fit_transform(X).astype(np.float32)
    Xtf = pre_full.transform(X_test).astype(np.float32)
    joblib.dump(pre_full, os.path.join(MODELS, "preprocessor.joblib"))

    logreg_full = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xfull, y)
    joblib.dump(logreg_full, os.path.join(MODELS, "logreg.joblib"))
    hgb_full = make_hgb().fit(Xfull, y)
    joblib.dump(hgb_full, os.path.join(MODELS, "hgb.joblib"))

    test_probs = {
        "Logistic Regression (baseline)": logreg_full.predict_proba(Xtf)[:, 1],
        "Gradient Boosting (baseline)": hgb_full.predict_proba(Xtf)[:, 1],
    }
    mlp_test = []
    for seed in MLP_SEEDS:
        m = fit_full_mlp(Xfull, y, seed, max(5, mlp_epochs[seed]))
        torch.save(m.state_dict(), os.path.join(MODELS, f"mlp_seed{seed}.pt"))
        mlp_test.append(predict_mlp(m, Xtf))
    test_probs["Deep Learning MLP (seed ensemble)"] = np.mean(mlp_test, axis=0)
    test_probs["Ensemble (MLP + HGB)"] = (
        test_probs["Deep Learning MLP (seed ensemble)"]
        + test_probs["Gradient Boosting (baseline)"]) / 2
    test_probs["Ensemble (MLP + HGB + LogReg)"] = (
        test_probs["Deep Learning MLP (seed ensemble)"]
        + test_probs["Gradient Boosting (baseline)"]
        + test_probs["Logistic Regression (baseline)"]) / 3

    with open(os.path.join(MODELS, "mlp_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"n_in": int(Xfull.shape[1]), "seeds": MLP_SEEDS,
                   "best_epochs": {str(s): int(e) for s, e in mlp_epochs.items()},
                   "full_epochs": {str(s): int(max(5, e)) for s, e in mlp_epochs.items()}},
                  fh, indent=2)

    val_df = pd.DataFrame({"customer_id": val_ids, "target": y_va})
    for name, p in probs.items():
        val_df[name] = p
    val_df.to_csv(os.path.join(REPORTS, "validation_predictions.csv"), index=False)

    test_prob_df = pd.DataFrame({"customer_id": test["customer_id"].to_numpy()})
    for name, p in test_probs.items():
        test_prob_df[name] = p
    test_prob_df["selected_probability"] = test_probs[best_name]
    test_prob_df.to_csv(os.path.join(REPORTS, "test_probabilities.csv"), index=False)

    submission = pd.DataFrame({
        "customer_id": test["customer_id"].to_numpy(),
        "target": (test_probs[best_name] >= best_thr).astype(int),
    })
    submission.to_csv(os.path.join(OUT_OUTPUTS, "submission.csv"), index=False)

    comp = pd.DataFrame([{
        "model": r["model"],
        "roc_auc": r["at_best_f1"]["roc_auc"],
        "pr_auc": r["at_best_f1"]["pr_auc"],
        "f1": r["at_best_f1"]["f1"],
        "precision": r["at_best_f1"]["precision"],
        "recall": r["at_best_f1"]["recall"],
        "threshold": r["at_best_f1"]["threshold"],
        "log_loss": r["at_best_f1"]["log_loss"],
        "brier": r["at_best_f1"]["brier"],
    } for r in results])
    comp.to_csv(os.path.join(REPORTS, "model_comparison.csv"), index=False)
    pd.DataFrame(mlp_log).to_csv(os.path.join(REPORTS, "mlp_training_log.csv"), index=False)

    out = {
        "seed": SEED, "mlp_seeds": MLP_SEEDS,
        "split": {"train": len(X_tr), "validation": len(X_va), "test": len(X_test)},
        "positive_rate_train": round(float(y.mean()), 5),
        "transformed_width": int(Xtr.shape[1]),
        "selected_model": best_name,
        "selected_threshold": best_thr,
        "models": results,
        "submission_positive_rate": round(float(submission["target"].mean()), 5),
        "submission_positives": int(submission["target"].sum()),
    }
    with open(os.path.join(REPORTS, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"\nSaved models, metrics and submission. Positives: {submission['target'].sum():,} "
          f"({submission['target'].mean():.1%})")


if __name__ == "__main__":
    main()
