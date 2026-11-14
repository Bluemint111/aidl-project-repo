from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from model import build_feature_table, train_and_score


DATA_DIR = Path(__file__).resolve().parent / "ShopSphere_DataPack"

st.set_page_config(
    page_title="ShopSphere Purchase Propensity",
    page_icon="🛍️",
    layout="wide",
)


@st.cache_data(show_spinner="Reading ShopSphere history and building features...")
def load_features(data_dir: str) -> tuple[pd.DataFrame, pd.Timestamp]:
    return build_feature_table(data_dir)


def gemini_insights(
    api_key: str,
    metrics: dict[str, float | int],
    scores: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> str:
    from google import genai

    top_count = max(1, math.ceil(len(scores) * 0.1))
    summary = {
        "prediction_horizon_days": 30,
        "historical_data_cutoff": as_of_date.strftime("%Y-%m-%d"),
        "scored_customers": int(len(scores)),
        "validation_metrics": metrics,
        "mean_score_all_customers": float(scores["purchase_probability_30d"].mean()),
        "mean_score_top_10_percent": float(
            scores.head(top_count)["purchase_probability_30d"].mean()
        ),
    }
    prompt = (
        "Interpret these aggregate validation and scoring statistics for a supervised "
        "customer purchase-propensity model. Suggest concise, cautious next steps for "
        "a CRM analyst. Distinguish ranking quality from calibrated probability; do "
        "not claim causality or recommend contacting customers without consent and "
        "policy checks. No customer-level data is included.\n\n"
        f"Aggregate statistics: {summary}"
    )
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty insight response.")
    return response.text


st.title("ShopSphere: 30-day purchase propensity")
st.write(
    "Train a supervised multilayer perceptron on the labeled customer set, validate "
    "its ranking performance, then score the unlabeled customers. Probabilities are "
    "estimates for prioritization, not guarantees."
)
st.caption(
    "Training and scoring run locally. Gemini is optional and receives only aggregate "
    "model statistics, never customer-level data."
)

with st.sidebar:
    st.header("Scoring settings")
    priority_percent = st.slider(
        "Mark highest-scoring customers as priority",
        min_value=1,
        max_value=50,
        value=10,
        format="%d%%",
    )
    gemini_key = st.text_input(
        "Gemini API key (optional)",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="Used only to summarize aggregate validation and scoring statistics.",
    )

if st.button("Train model and score test customers", type="primary"):
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        st.error(f"Expected train.csv and test.csv in {DATA_DIR}")
    else:
        with st.spinner("Building features and training the neural network..."):
            feature_table, as_of_date = load_features(str(DATA_DIR))
            train = pd.read_csv(
                train_path, encoding="utf-8-sig", dtype={"customer_id": "string"}
            )
            test = pd.read_csv(
                test_path, encoding="utf-8-sig", dtype={"customer_id": "string"}
            )
            if not {"customer_id", "target"}.issubset(train.columns):
                raise ValueError("train.csv must contain customer_id and target.")
            if "customer_id" not in test.columns:
                raise ValueError("test.csv must contain customer_id.")
            labels = train.set_index("customer_id")["target"]
            scores, metrics = train_and_score(
                feature_table, labels, test["customer_id"]
            )
        st.session_state["shop_scores"] = scores
        st.session_state["shop_metrics"] = metrics
        st.session_state["shop_as_of_date"] = as_of_date

if "shop_scores" in st.session_state:
    scores = st.session_state["shop_scores"]
    metrics = st.session_state["shop_metrics"]
    as_of_date = st.session_state["shop_as_of_date"]
    top_count = max(1, math.ceil(len(scores) * priority_percent / 100))

    st.subheader("Validation")
    st.caption(
        f"Historical features through {as_of_date:%Y-%m-%d}. Validation uses a "
        "stratified 20% holdout from train.csv; the scoring model is then refit on "
        "all labeled customers."
    )
    metric_columns = st.columns(5)
    metric_columns[0].metric("ROC-AUC", f"{metrics['validation_roc_auc']:.3f}")
    metric_columns[1].metric(
        "Average precision", f"{metrics['validation_average_precision']:.3f}"
    )
    metric_columns[2].metric(
        "Precision in top 10%",
        f"{metrics['precision_at_top_10_percent']:.1%}",
    )
    metric_columns[3].metric(
        "Recall in top 10%",
        f"{metrics['recall_at_top_10_percent']:.1%}",
    )
    metric_columns[4].metric(
        "Validation positive rate", f"{metrics['validation_positive_rate']:.1%}"
    )
    st.caption(
        f"Model training stopped after {metrics['training_iterations']} iterations. "
        "Probability scores are ranking estimates and are not calibrated guarantees."
    )

    st.subheader(f"Ranked test customers — top {priority_percent}% highlighted")
    result = scores.copy()
    result["priority"] = result["rank"].le(top_count)
    result["purchase_probability_30d"] = result[
        "purchase_probability_30d"
    ].round(4)
    st.dataframe(result.head(max(top_count, 100)), use_container_width=True)
    st.download_button(
        "Download all customer scores",
        data=result.to_csv(index=False),
        file_name="shopsphere_purchase_propensity_30d.csv",
        mime="text/csv",
    )

    if st.button("Generate aggregate Gemini insights"):
        if not gemini_key.strip():
            st.warning("Enter a Gemini API key in the sidebar first.")
        else:
            with st.spinner("Requesting aggregate-only insights from Gemini..."):
                st.markdown(
                    gemini_insights(
                        gemini_key.strip(), metrics, scores, as_of_date
                    )
                )

st.divider()
st.caption(
    "Use this score alongside consent, suppression, support-case, and contact-frequency "
    "policies before any outreach. This model predicts association, not the effect of "
    "a marketing action."
)
