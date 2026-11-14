"""
ShopSphere Propensity Studio - sample-level segment explorer and tester.

The dataset is treated as a sample of the customer base: every view aggregates
customers into segments. No customer IDs or individual lookup are used.

Run from the project root:

    py -m streamlit run archive/legacy_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)

BASE = Path(__file__).resolve().parents[1]
FEAT = BASE / "artifacts" / "features"
REPORTS = BASE / "artifacts" / "reports" / "fixed"
DATA = BASE / "data" / "clean"

st.set_page_config(page_title="ShopSphere Segment Studio", layout="wide")


@st.cache_data
def load_json(name: str) -> dict:
    with open(REPORTS / name, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def load_report(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS / name)


@st.cache_data
def load_features(split: str) -> pd.DataFrame:
    return pd.read_pickle(FEAT / f"{split}_features.pkl")


@st.cache_data
def load_data_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


@st.cache_data
def data_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(DATA.glob("*.csv")):
        if path.name == "sample_submission.csv":
            continue
        with open(path, encoding="utf-8", errors="ignore") as fh:
            n = sum(1 for _ in fh) - 1
        rows.append({"table": path.stem, "rows": n})
    return pd.DataFrame(rows)


def _quartile(s: pd.Series, labels: list[str]) -> pd.Series:
    return pd.qcut(s.rank(method="first"), 4, labels=labels)


@st.cache_data
def segment_frame(split: str) -> pd.DataFrame:
    """Customer rows joined to out-of-sample scores, then summarised by segment.

    customer_id is dropped immediately: all downstream use is group-level.
    """
    feats = load_features("train" if split == "validation" else split)
    cols = ["customer_id", "cust_country", "cust_region", "cust_tier", "pref_channel",
            "pref_language", "pref_email_opt", "pref_sms_opt", "pref_dnc", "sup_n",
            "sup_open_n", "sup_recent_complaint", "ret_days_since_last", "cust_loyalty",
            "tenure_days", "sess_n", "sess_n_180d", "sess_days_since_last",
            "txn_spend_180d", "txn_days_since_last"]
    out = feats[cols].copy()
    if split == "test":
        p = load_report("test_probabilities.csv")[["customer_id", "selected_probability"]]
        out = out.merge(p, on="customer_id", how="left")
        out["target"] = np.nan
    else:
        p = load_report("validation_predictions.csv")[["customer_id", "target", SEL]]
        p = p.rename(columns={SEL: "selected_probability"})
        out = out.merge(p, on="customer_id", how="left")
    out = out[out["selected_probability"].notna()].reset_index(drop=True)
    out = out.drop(columns=["customer_id"])

    b = pd.DataFrame(index=out.index)
    b["tenure_band"] = pd.cut(
        out["tenure_days"], [-1, 90, 365, 1095, 1e9],
        labels=["New <90d", "Developing 90d-1y", "Established 1-3y", "Loyal 3y+"])
    rec = out["sess_days_since_last"]
    b["recency_band"] = np.select(
        [rec.isna(), rec <= 7, rec <= 30, rec <= 90],
        ["No session", "Active <=7d", "Warm 8-30d", "Cooling 31-90d"],
        default="Dormant >90d")
    txn = out["txn_days_since_last"]
    b["txn_recency_band"] = np.select(
        [txn.isna(), txn <= 30, txn <= 90, txn <= 180],
        ["No purchase", "Bought <=30d", "31-90d", "91-180d"], default="181d+")
    b["loyalty_band"] = _quartile(out["cust_loyalty"], ["Q1 low", "Q2", "Q3", "Q4 high"])
    b["session_band"] = _quartile(out["sess_n_180d"], ["Q1 low", "Q2", "Q3", "Q4 high"])
    b["spend_band"] = _quartile(out["txn_spend_180d"], ["Q1 low", "Q2", "Q3", "Q4 high"])
    b["support_band"] = np.select(
        [out["sup_open_n"].fillna(0) > 0, out["sup_recent_complaint"].fillna(0) == 1,
         out["sup_n"].fillna(0) > 0],
        ["Open complaint", "Recent complaint (resolved)", "Past issue"],
        default="No recent issue")
    return pd.concat([out, b], axis=1)


DIMENSIONS = {
    "Tier": "cust_tier",
    "Country": "cust_country",
    "Region": "cust_region",
    "Preferred channel": "pref_channel",
    "Language": "pref_language",
    "Tenure band": "tenure_band",
    "Session recency": "recency_band",
    "Purchase recency": "txn_recency_band",
    "Loyalty band": "loyalty_band",
    "Session volume": "session_band",
    "Spend band": "spend_band",
    "Support status": "support_band",
}


def agg(frame: pd.DataFrame, by: list[str], thr: float) -> pd.DataFrame:
    total = max(float(frame["selected_probability"].sum()), 1e-12)
    group = frame.groupby(by, observed=True)
    out = group.agg(
        customers=("selected_probability", "size"),
        mean_propensity=("selected_probability", "mean"),
        expected_buyers=("selected_probability", "sum"),
        flag_rate=("selected_probability", lambda s: float((s >= thr).mean())),
        actual_buyer_rate=("target", "mean"),
    ).reset_index()
    if not frame["target"].notna().any():
        out = out.drop(columns=["actual_buyer_rate"])
    out["share_expected"] = out["expected_buyers"] / total
    return out.sort_values("expected_buyers", ascending=False)


def preset_segments(frame: pd.DataFrame, thr: float) -> pd.DataFrame:
    total = max(float(frame["selected_probability"].sum()), 1e-12)
    defs = {
        "Bronze + active + high sessions": (
            (frame["cust_tier"] == "Bronze")
            & (frame["recency_band"] == "Active <=7d")
            & (frame["session_band"] == "Q4 high")),
        "Email opt-in + high sessions": (
            (frame["pref_email_opt"] == 1) & (frame["session_band"] == "Q4 high")),
        "Bronze + email opt-in": (
            (frame["cust_tier"] == "Bronze") & (frame["pref_email_opt"] == 1)),
        "High loyalty + cooling": (
            (frame["loyalty_band"] == "Q4 high")
            & (frame["recency_band"] == "Cooling 31-90d")),
        "Gold/Silver + recent complaint": (
            frame["cust_tier"].isin(["Gold", "Silver"])
            & (frame["sup_recent_complaint"] == 1)),
        "High spend + dormant": (
            (frame["spend_band"] == "Q4 high")
            & (frame["recency_band"] == "Dormant >90d")),
        "Platinum + dormant": (
            (frame["cust_tier"] == "Platinum")
            & (frame["recency_band"] == "Dormant >90d")),
        "Open complaint (all tiers)": frame["sup_open_n"].fillna(0) > 0,
        "DNC customers": frame["pref_dnc"] == 1,
    }
    rows = []
    for name, mask in defs.items():
        s = frame.loc[mask, "selected_probability"]
        row = {"segment": name, "customers": len(s),
               "mean_propensity": s.mean(), "expected_buyers": s.sum(),
               "share_expected": s.sum() / total,
               "flag_rate": float((s >= thr).mean()) if len(s) else np.nan}
        if frame["target"].notna().any():
            row["actual_buyer_rate"] = frame.loc[mask, "target"].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("expected_buyers", ascending=False)


def draw_heatmap(pivot: pd.DataFrame, ax, title: str, fmt: str, cmap: str) -> None:
    data = pivot.to_numpy(dtype=float)
    ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                        fontsize=8, color="black")
    ax.set_title(title, fontsize=10)
    ax.grid(False)


def style_axes(ax):
    ax.grid(alpha=0.25, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def fmt_pct(x: float) -> str:
    return f"{x:.1%}"


METRICS = load_json("metrics.json")
COMP = load_report("model_comparison.csv")
VAL = load_report("validation_predictions.csv")
SEL = METRICS["selected_model"]
THR = float(METRICS["selected_threshold"])
MODEL_COLS = [c for c in VAL.columns if c not in ("customer_id", "target")]


def risk_band(p: float, thr: float) -> str:
    if p >= thr:
        return "High"
    if p >= thr / 2:
        return "Medium"
    return "Low"


st.title("ShopSphere Segment Studio")
st.caption(
    "Sample-level propensity analysis - data cutoff 2024-09-30. The dataset is treated "
    "as a sample of the customer base, so all views are segment estimates for planning, "
    "not individual decisions."
)
st.info(
    "This tool deliberately avoids customer IDs and individual lookup. It shows how "
    "propensity, expected buyers and budgeted outreach differ across groups - including "
    "customers who sit in several groups at once (tier, recency, loyalty, support, channel)."
)

tab_overview, tab_models, tab_segments, tab_planner = st.tabs(
    ["Overview", "Model results", "Segment explorer", "Win-back segments"]
)

# --------------------------------------------------------------------------- overview
with tab_overview:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Training sample", f"{METRICS['split']['train']:,}")
    c2.metric("Validation sample", f"{METRICS['split']['validation']:,}")
    c3.metric("Test sample", f"{METRICS['split']['test']:,}")
    best = next(m for m in METRICS["models"] if m["model"] == SEL)
    c4.metric("Validation PR-AUC", f"{best['at_best_f1']['pr_auc']:.4f}")
    c5.metric("Validation F1", f"{best['at_best_f1']['f1']:.4f}")
    c6.metric("Decision threshold", f"{THR:.3f}")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("How to read this")
        st.markdown(
            f"""
- The data is a **sample**, not the full customer base. Figures are estimates of what
  the wider population in each segment might do.
- Models compare a deep MLP, non-deep baselines and blended ensembles on a held-out
  sample. Selected model: **{SEL}** at threshold **{THR:.3f}**.
- The deep model on its own reaches PR-AUC
  {next(m for m in METRICS['models'] if m['model'].startswith('Deep'))['at_best_f1']['pr_auc']:.4f}
  and improves when blended with gradient boosting.
- The **Segment explorer** aggregates customers by one or several dimensions; the
  **Win-back segments** tab applies consent, complaint and budget policy then reports the
  plan at segment level.
            """
        )
        st.subheader("Sample inventory")
        inv = data_inventory()
        st.dataframe(inv, width="stretch", hide_index=True, height=230)
        st.caption(f"{len(inv)} cleaned tables, {inv['rows'].sum():,} total rows.")

        st.subheader("Segment snapshot by tier (test sample)")
        tier = agg(segment_frame("test"), ["cust_tier"], THR).rename(columns={"cust_tier": "Tier"})
        st.dataframe(
            tier[["Tier", "customers", "share_expected", "mean_propensity",
                  "expected_buyers", "flag_rate"]],
            width="stretch", hide_index=True,
            column_config={
                "mean_propensity": st.column_config.NumberColumn(format="%.3f"),
                "expected_buyers": st.column_config.NumberColumn(format="%.0f"),
                "share_expected": st.column_config.NumberColumn(format="%.3f"),
                "flag_rate": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption(
            "Bronze is the largest group but a small share of expected buyers; Platinum is "
            "small but has the highest propensity. Segment views below add the intersections."
        )
    with right:
        st.subheader("Validation precision-recall")
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for name in ["Deep Learning MLP (seed ensemble)", "Gradient Boosting (baseline)",
                     "Logistic Regression (baseline)", "Ensemble (MLP + HGB)"]:
            if name in VAL:
                prec, rec, _ = precision_recall_curve(VAL["target"], VAL[name])
                ap = average_precision_score(VAL["target"], VAL[name])
                ax.plot(rec, prec, linewidth=1.8, label=f"{name}  AP={ap:.4f}")
        base = VAL["target"].mean()
        ax.axhline(base, color="#777", linestyle="--", linewidth=1, label=f"Prior {base:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7, loc="upper right")
        style_axes(ax)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        st.subheader("Test sample snapshot")
        sub = load_report("test_probabilities.csv")
        n_pos = int((sub["selected_probability"] >= THR).sum())
        a, b, c = st.columns(3)
        a.metric("Sample customers", f"{len(sub):,}")
        b.metric("Expected buyers", f"{sub['selected_probability'].sum():,.0f}")
        c.metric("Above threshold", fmt_pct(n_pos / len(sub)))

    with st.expander("Responsible AI and model card"):
        st.markdown(
            f"""
- **Task:** probability a customer purchases in the 30 days after 2024-09-30.
- **Features:** {METRICS['transformed_width']} transformed columns from 12 behavioural
  tables; no name/email/phone identifiers are used.
- **Sample, not census:** results estimate segment behaviour. Do not treat a segment
  estimate as a decision about any one customer.
- **Selection metric:** PR-AUC (class imbalance {METRICS['positive_rate_train']:.2%});
  threshold tuned on validation F1.
- **Leakage controls:** all features use data on/before the cutoff; test customers are
  disjoint from train.
- **Intended use:** prioritise segments for retention experiments. Propensity is not
  causal uplift - run a randomised holdout before scaling incentives.
- **Fairness:** selection rate is monitored by country, tier and intersections. Keep
  human review for high-value or complaint-related segments.
            """
        )

# --------------------------------------------------------------------------- models
with tab_models:
    st.subheader("Model leaderboard (validation sample)")
    show = COMP.sort_values("pr_auc", ascending=False).copy()
    show.insert(0, "selected", np.where(show["model"] == SEL, "yes", ""))
    st.dataframe(
        show[["selected", "model", "roc_auc", "pr_auc", "f1", "precision", "recall",
              "threshold", "log_loss", "brier"]],
        width="stretch", hide_index=True,
        column_config={
            "roc_auc": st.column_config.NumberColumn(format="%.4f"),
            "pr_auc": st.column_config.NumberColumn(format="%.4f"),
            "f1": st.column_config.NumberColumn(format="%.4f"),
            "precision": st.column_config.NumberColumn(format="%.4f"),
            "recall": st.column_config.NumberColumn(format="%.4f"),
            "threshold": st.column_config.NumberColumn(format="%.3f"),
            "log_loss": st.column_config.NumberColumn(format="%.4f"),
            "brier": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    left, right = st.columns(2)
    with left:
        st.subheader("ROC curves")
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        for name in MODEL_COLS:
            fpr, tpr, _ = roc_curve(VAL["target"], VAL[name])
            ax.plot(fpr, tpr, linewidth=1.7,
                    label=f"{name}  {roc_auc_score(VAL['target'], VAL[name]):.4f}")
        ax.plot([0, 1], [0, 1], color="#999", linestyle=":", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(fontsize=6, loc="lower right")
        style_axes(ax)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with right:
        st.subheader("Threshold tester")
        model = st.selectbox("Model", MODEL_COLS, index=MODEL_COLS.index(SEL))
        thr = st.slider("Decision threshold", 0.05, 0.95, float(THR), 0.01)
        prob = VAL[model].to_numpy()
        y = VAL["target"].to_numpy()
        pred = (prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{prec:.3f}")
        m2.metric("Recall", f"{rec:.3f}")
        m3.metric("F1", f"{f1:.3f}")
        m4.metric("Flagged", f"{int(pred.sum()):,}")
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        ConfusionMatrixDisplay(np.array([[tn, fp], [fn, tp]]),
                               display_labels=["No buy", "Buy"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("")
        st.pyplot(fig, width="content")
        plt.close(fig)

    st.subheader("What drives the model (permutation importance, gradient boosting)")
    imp = load_report("feature_importance_hgb.csv").head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(imp["feature"], imp["importance"], color="#2c6fbb")
    ax.set_xlabel("Drop in PR-AUC when shuffled")
    style_axes(ax)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    with st.expander("Deep-learning training curves (3 seeds)"):
        log = load_report("mlp_training_log.csv")
        pivot = log.pivot_table(index="epoch", columns="seed", values="val_ap")
        st.line_chart(pivot)
        st.caption("Validation PR-AUC per epoch; early stopping restored each seed's best epoch.")

    with st.expander("Fairness snapshot - selection rate by group (test sample)"):
        frame = segment_frame("test").copy()
        frame["flagged"] = frame["selected_probability"] >= THR
        for key in ("cust_country", "cust_tier", "recency_band"):
            table = (frame.groupby(key, observed=True)
                     .agg(customers=("flagged", "size"), flagged=("flagged", "sum"),
                          selection_rate=("flagged", "mean"))
                     .reset_index().sort_values("selection_rate", ascending=False))
            st.markdown(f"**By {key.replace('cust_', '').replace('_', ' ')}**")
            st.dataframe(table, width="stretch", hide_index=True,
                         column_config={"selection_rate": st.column_config.NumberColumn(format="%.3f")})

# --------------------------------------------------------------------------- segments
with tab_segments:
    st.subheader("Group and multi-group analysis")
    st.caption(
        "Every row is an aggregate of customers. Build a segment with filters, then group "
        "by up to three dimensions to see who lands in multiple groups at once."
    )

    split_label = st.radio(
        "Dataset", ["Test sample (unseen)", "Validation sample (known outcomes)"],
        horizontal=True)
    split = "test" if split_label.startswith("Test") else "validation"
    frame = segment_frame(split)
    thr = st.slider("Flag threshold", 0.05, 0.95, float(THR), 0.01, key="seg_thr")

    with st.expander("Filter the sample", expanded=False):
        f1, f2, f3, f4 = st.columns(4)
        tiers = sorted(frame["cust_tier"].dropna().unique())
        countries = sorted(frame["cust_country"].dropna().unique())
        channels = sorted(frame["pref_channel"].dropna().unique())
        recencies = ["Active <=7d", "Warm 8-30d", "Cooling 31-90d",
                     "Dormant >90d", "No session"]
        sel_tiers = f1.multiselect("Tier", tiers, default=tiers)
        sel_countries = f2.multiselect("Country", countries, default=countries)
        sel_channels = f3.multiselect("Preferred channel", channels, default=channels)
        sel_recency = f4.multiselect("Session recency", recencies,
                                     default=[r for r in recencies if r in frame["recency_band"].unique()])

    sel_tiers = sel_tiers or tiers
    sel_countries = sel_countries or countries
    sel_channels = sel_channels or channels
    sel_recency = sel_recency or recencies

    f = frame[
        frame["cust_tier"].isin(sel_tiers)
        & frame["cust_country"].isin(sel_countries)
        & frame["pref_channel"].isin(sel_channels)
        & frame["recency_band"].isin(sel_recency)
    ].copy()

    dims = st.multiselect(
        "Group by (up to 3)", list(DIMENSIONS), default=["Tier"],
        help="Pick several dimensions to see intersection groups.")
    dims = dims[:3] or ["Tier"]
    by = [DIMENSIONS[d] for d in dims]
    grouped = agg(f, by, thr)

    a, b, c, d = st.columns(4)
    a.metric("Customers in view", f"{len(f):,}")
    b.metric("Mean propensity", f"{f['selected_probability'].mean():.3f}")
    c.metric("Expected buyers", f"{f['selected_probability'].sum():,.0f}")
    d.metric("Share of sample expected", fmt_pct(f["selected_probability"].sum()
                                                  / max(frame["selected_probability"].sum(), 1e-12)))

    rename = {DIMENSIONS[d]: d for d in dims}
    shown = grouped.rename(columns=rename)
    show_cols = dims + ["customers", "mean_propensity", "expected_buyers",
                        "share_expected", "flag_rate"]
    if "actual_buyer_rate" in shown:
        show_cols.append("actual_buyer_rate")
    st.dataframe(
        shown[show_cols], width="stretch", hide_index=True, height=340,
        column_config={
            "mean_propensity": st.column_config.ProgressColumn(
                "mean propensity", min_value=0.0, max_value=1.0, format="%.3f"),
            "expected_buyers": st.column_config.NumberColumn(format="%.0f"),
            "share_expected": st.column_config.NumberColumn(format="%.3f"),
            "flag_rate": st.column_config.NumberColumn(format="%.3f"),
            "actual_buyer_rate": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.download_button(
        "Download segment table (CSV)", shown[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="segment_summary.csv", mime="text/csv",
    )

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Top segments by expected buyers")
        top = shown.sort_values("expected_buyers", ascending=False).head(12).iloc[::-1].copy()
        top["label"] = top[dims].astype(str).agg(" | ".join, axis=1)
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.barh(top["label"], top["expected_buyers"], color="#2c6fbb")
        ax.set_xlabel("Expected buyers (sum of propensity)")
        style_axes(ax)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with right:
        st.subheader("Choose a heatmap view")
        hm_row = st.selectbox("Rows", list(DIMENSIONS), index=list(DIMENSIONS).index("Tier"))
        hm_col = st.selectbox("Columns", list(DIMENSIONS),
                              index=list(DIMENSIONS).index("Session recency"))
        if hm_row == hm_col:
            st.info("Choose two different dimensions for the heatmap.")
        else:
            r, c = DIMENSIONS[hm_row], DIMENSIONS[hm_col]
            p_mean = f.pivot_table(index=r, columns=c, values="selected_probability",
                                   aggfunc="mean")
            p_cnt = f.pivot_table(index=r, columns=c, values="selected_probability",
                                  aggfunc="size")
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
            draw_heatmap(p_mean, axes[0], f"Mean propensity: {hm_row} x {hm_col}",
                         "{:.2f}", "Blues")
            draw_heatmap(p_cnt.fillna(0), axes[1], f"Customers: {hm_row} x {hm_col}",
                         "{:.0f}", "Greens")
            plt.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)

    st.subheader("Multi-group spotlight (intersections)")
    st.caption(
        "Each row contains only customers who satisfy every condition at once - for "
        "example Bronze AND active AND high session volume. This is where a group can be "
        "large in headcount but small in expected buyers, or vice versa."
    )
    presets = preset_segments(f, thr)
    st.dataframe(
        presets, width="stretch", hide_index=True,
        column_config={
            "mean_propensity": st.column_config.NumberColumn(format="%.3f"),
            "expected_buyers": st.column_config.NumberColumn(format="%.0f"),
            "share_expected": st.column_config.NumberColumn(format="%.3f"),
            "flag_rate": st.column_config.NumberColumn(format="%.3f"),
            "actual_buyer_rate": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.download_button(
        "Download multi-group table (CSV)", presets.to_csv(index=False).encode("utf-8"),
        file_name="segment_intersections.csv", mime="text/csv",
    )

# --------------------------------------------------------------------------- planner
with tab_planner:
    st.subheader("Budgeted win-back plan by segment")
    st.caption(
        "Row-level policy is applied, then the plan is reported as segments. No customer "
        "IDs are shown or exported. This is the policy tool the agent will call."
    )
    budget = load_data_csv("retention_budget_2024Q4.csv")
    offers = load_data_csv("offer_catalogue.csv")
    frame = segment_frame("test")

    c1, c2, c3, c4 = st.columns(4)
    week = c1.selectbox("Week", sorted(budget["week_start"].unique()))
    country_pick = c2.selectbox("Country", ["All"] + sorted(budget["country"].unique()))
    thr = c3.slider("Minimum propensity", 0.05, 0.95, float(THR), 0.01, key="plan_thr")
    max_contacts = c4.slider("Max contacts", 50, 2000, 500, 50)

    c5, c6, c7, c8 = st.columns(4)
    aov = c5.number_input("Assumed order value (USD)", 20.0, 500.0, 80.0, 5.0)
    consent = c6.checkbox("Respect consent / DNC", value=True)
    complaints = c7.checkbox("Exclude open complaints", value=True)
    recent = c8.checkbox("Skip contacted < 14 days", value=True)

    def offer_for(row) -> str:
        if row.get("sup_recent_complaint", 0) == 1:
            return "OFF-SRC"
        return {"Platinum": "OFF-V20", "Gold": "OFF-V15",
                "Silver": "OFF-V15"}.get(row.get("cust_tier"), "OFF-FS")

    def offer_cost(code: str) -> float:
        rec = offers[offers["offer_code"] == code].iloc[0]
        return float(rec["value"]) * aov if rec["offer_type"] == "Voucher" else float(rec["value"])

    cand = frame[frame["selected_probability"] >= thr].copy()
    audit = [{"stage": "Above propensity threshold", "customers": len(cand)}]
    if consent:
        cand = cand[cand["pref_dnc"] == 0]
        audit.append({"stage": "After DNC / consent", "customers": len(cand)})
    if complaints:
        cand = cand[cand["sup_open_n"].fillna(0) == 0]
        audit.append({"stage": "After open-complaint suppression", "customers": len(cand)})
    if recent:
        cand = cand[cand["ret_days_since_last"].isna() | (cand["ret_days_since_last"] >= 14)]
        audit.append({"stage": "After recent-contact suppression", "customers": len(cand)})
    if country_pick != "All":
        cand = cand[cand["cust_country"] == country_pick]
        audit.append({"stage": f"After country filter ({country_pick})", "customers": len(cand)})

    cand["offer"] = cand.apply(offer_for, axis=1)
    cand["estimated_cost"] = cand["offer"].map(offer_cost)
    cand = cand.sort_values("selected_probability", ascending=False)

    budgets = budget[budget["week_start"] == week].set_index("country")["retention_budget_usd"]
    chosen = []
    for country, grp in cand.groupby("cust_country", observed=True):
        cap = float(budgets.get(country, 0))
        spent, take = 0.0, []
        for _, row in grp.sort_values("selected_probability", ascending=False).iterrows():
            cost = float(row["estimated_cost"])
            if spent + cost > cap:
                break
            take.append(row)
            spent += cost
            if len(take) >= max_contacts:
                break
        if take:
            chosen.append(pd.DataFrame(take))
    plan = (pd.concat(chosen, ignore_index=True) if chosen
            else cand.iloc[0:0].assign(estimated_cost=pd.Series(dtype=float)))
    if len(plan) > max_contacts:
        plan = plan.sort_values("selected_probability", ascending=False).head(max_contacts)
    plan = plan.sort_values("selected_probability", ascending=False)

    a, b, c, d = st.columns(4)
    a.metric("Eligible customers", f"{len(cand):,}")
    b.metric("Selected contacts", f"{len(plan):,}")
    c.metric("Expected buyers", f"{plan['selected_probability'].sum():.0f}")
    d.metric("Estimated spend", f"${plan['estimated_cost'].sum():,.0f}")
    budget_total = float(budgets.sum()) if country_pick == "All" else float(budgets.get(country_pick, 0))
    st.caption(f"Weekly budget for {country_pick}: ${budget_total:,.0f}. "
               f"Utilisation: {plan['estimated_cost'].sum() / max(budget_total, 1):.1%}.")

    plan_agg = (plan.groupby(["cust_country", "cust_tier", "offer", "pref_channel"],
                             observed=True)
                .agg(contacts=("selected_probability", "size"),
                     expected_buyers=("selected_probability", "sum"),
                     estimated_spend=("estimated_cost", "sum"))
                .reset_index()
                .sort_values("expected_buyers", ascending=False))
    st.dataframe(
        plan_agg, width="stretch", hide_index=True, height=340,
        column_config={
            "expected_buyers": st.column_config.NumberColumn(format="%.1f"),
            "estimated_spend": st.column_config.NumberColumn(format="$%.2f"),
        },
    )
    st.download_button(
        "Download segment plan (CSV)", plan_agg.to_csv(index=False).encode("utf-8"),
        file_name="winback_segment_plan.csv", mime="text/csv",
    )

    with st.expander("Policy funnel and audit trail"):
        st.dataframe(pd.DataFrame(audit), width="stretch", hide_index=True)
        st.markdown(
            "Offer policy: Platinum -> 20% voucher, Silver/Gold -> 15% voucher, "
            "Bronze -> free shipping, recent resolved complaint -> service-recovery credit. "
            "Voucher cost = value x assumed order value. Before launch, run a randomised "
            "segment holdout to measure incremental uplift."
        )

st.divider()
st.caption("ShopSphere - IT2226 AI Innovation & Deep Learning. Segment-level analysis; "
           "no customer IDs or individual targeting decisions.")
