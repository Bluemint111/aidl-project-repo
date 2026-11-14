from __future__ import annotations

import json
import os

import altair as alt
import pandas as pd
import streamlit as st

from shopsphere.app import approvals
from shopsphere.common import FIXED_REPORTS, OUT_DATA, ROLLING_REPORTS, read_budget
from shopsphere.pipelines import planner

st.set_page_config(
    page_title="ShopSphere Retention Intelligence",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(OUT_DATA, "offer_candidate_scores.parquet"))


def analytics(name: str) -> pd.DataFrame:
    path = os.path.join(OUT_DATA, name)
    if not os.path.exists(path):
        st.error(f"Missing analytics file: {path}. Run build_analytics.py first.")
        st.stop()
    return load_csv(path)


def fmt_money(value) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"${float(value):,.0f}"
    except Exception:
        return "n/a"


def fmt_pct(value) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.1%}"
    except Exception:
        return "n/a"


def show_table(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    out = df[columns].copy() if columns else df.copy()
    st.dataframe(out, width="stretch", hide_index=True)


def tab_overview() -> None:
    st.subheader("Executive performance")
    daily = analytics("analytics_daily_country.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    countries = sorted(daily["country"].dropna().unique())
    max_date = daily["date"].max().date()
    min_date = daily["date"].min().date()

    c1, c2, c3 = st.columns([1.1, 1.1, 2])
    date_range = c1.date_input(
        "Date range", value=(max_date - pd.Timedelta(days=89), max_date),
        min_value=min_date, max_value=max_date,
    )
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start, end = date_range
    else:
        start = end = date_range
    selected = c2.multiselect("Countries", countries, default=countries)
    metric = c3.selectbox(
        "Trend metric", ["revenue", "orders", "sessions", "aov", "conversion_rate"],
        format_func=lambda x: {
            "revenue": "Revenue", "orders": "Orders", "sessions": "Sessions",
            "aov": "AOV", "conversion_rate": "Conversion rate",
        }[x],
    )
    d = daily[
        (daily["date"] >= pd.Timestamp(start)) & (daily["date"] <= pd.Timestamp(end)) &
        (daily["country"].isin(selected))
    ].copy()
    if d.empty:
        st.warning("No rows in the selected window.")
        return

    revenue = d["revenue"].sum()
    orders = d["orders"].sum()
    sessions = d["sessions"].sum()
    spend = d["campaign_spend"].sum()
    attr_rev = d["attributed_revenue"].sum()
    deliveries = d["deliveries"].sum()
    late = d["late_deliveries"].sum()
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Revenue", fmt_money(revenue))
    k2.metric("Orders", f"{orders:,.0f}")
    k3.metric("AOV", fmt_money(revenue / orders if orders else None))
    k4.metric("Conversion", fmt_pct(orders / sessions if sessions else None))
    k5.metric("Campaign ROAS", f"{attr_rev / spend:.2f}" if spend else "n/a")
    k6.metric("Late delivery", fmt_pct(late / deliveries if deliveries else None))

    chart = alt.Chart(d).mark_line(point=False).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y(f"{metric}:Q", title=metric.replace("_", " ").title()),
        color=alt.Color("country:N", title="Country"),
        tooltip=["date:T", "country:N", alt.Tooltip(f"{metric}:Q", format=".3f")],
    ).properties(height=320)
    st.altair_chart(chart, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**Country scorecard: last 90 days vs prior 90 days**")
        score = analytics("analytics_country_scorecard.csv")
        score = score[score["country"].isin(selected)]
        show = score[[
            "country", "revenue_last90", "revenue_change_pct", "orders_last90",
            "orders_change_pct", "aov_last90", "conversion_last90", "roas_last90",
            "late_rate_last90",
        ]].copy()
        show.columns = [
            "Country", "Revenue", "Revenue Δ", "Orders", "Orders Δ", "AOV",
            "Conversion", "ROAS", "Late rate",
        ]
        for col in ["Revenue", "AOV"]:
            show[col] = show[col].map(fmt_money)
        for col in ["Revenue Δ", "Orders Δ", "Conversion", "Late rate"]:
            show[col] = show[col].map(fmt_pct)
        show["ROAS"] = show["ROAS"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a")
        show_table(show)
    with right:
        st.markdown("**Carrier reliability**")
        carrier = analytics("analytics_carrier.csv")
        carrier = carrier[carrier["country"].isin(selected)]
        show = carrier[[
            "country", "carrier", "deliveries", "late_rate", "lost_rate",
            "returned_rate", "avg_delay_days",
        ]].copy()
        show.columns = ["Country", "Carrier", "Deliveries", "Late", "Lost", "Returned", "Avg delay"]
        for col in ["Late", "Lost", "Returned"]:
            show[col] = show[col].map(fmt_pct)
        show["Avg delay"] = show["Avg delay"].map(
            lambda x: f"{x:.2f}d" if pd.notna(x) else "n/a"
        )
        show_table(show.sort_values(["Country", "Late"], ascending=[True, False]))


def tab_campaigns() -> None:
    st.subheader("Campaign effectiveness and incrementality")
    st.caption(
        "Attribution metrics are platform-attributed. The DiD view is a quasi-experimental "
        "region comparison and should not be read as a randomised causal estimate."
    )
    monthly = analytics("analytics_campaign_type.csv")
    monthly["month"] = pd.to_datetime(monthly["month"])
    types = sorted(monthly["campaign_type"].dropna().unique())
    selected = st.multiselect("Campaign types", types, default=types)
    m = monthly[monthly["campaign_type"].isin(selected)].copy()
    if m.empty:
        st.warning("No campaign rows.")
        return

    totals = m.groupby("campaign_type").agg(
        campaigns=("campaigns", "sum"), spend=("spend", "sum"),
        revenue=("attributed_revenue", "sum"), orders=("attributed_orders", "sum"),
        clicks=("clicks", "sum"), impressions=("impressions", "sum"),
    ).reset_index()
    totals["roas"] = totals["revenue"] / totals["spend"].replace(0, float("nan"))
    totals["cpa"] = totals["spend"] / totals["orders"].replace(0, float("nan"))
    totals["ctr"] = totals["clicks"] / totals["impressions"].replace(0, float("nan"))
    totals = totals.sort_values("roas", ascending=False)

    c1, c2 = st.columns([1, 1])
    with c1:
        bar = alt.Chart(totals).mark_bar().encode(
            x=alt.X("campaign_type:N", sort="-y", title=None),
            y=alt.Y("roas:Q", title="Attributed ROAS"),
            color=alt.Color("campaign_type:N", legend=None),
            tooltip=["campaign_type:N", alt.Tooltip("roas:Q", format=".2f"),
                     alt.Tooltip("spend:Q", format="$,.0f")],
        ).properties(height=300)
        st.altair_chart(bar, width="stretch")
    with c2:
        show = totals[["campaign_type", "spend", "revenue", "orders", "roas", "cpa", "ctr"]].copy()
        show.columns = ["Type", "Spend", "Revenue", "Orders", "ROAS", "CPA", "CTR"]
        show["Spend"] = show["Spend"].map(fmt_money)
        show["Revenue"] = show["Revenue"].map(fmt_money)
        show["ROAS"] = show["ROAS"].map(lambda x: f"{x:.2f}")
        show["CPA"] = show["CPA"].map(fmt_money)
        show["CTR"] = show["CTR"].map(fmt_pct)
        show_table(show)

    trend = alt.Chart(m).mark_line(point=False).encode(
        x=alt.X("month:T", title=None),
        y=alt.Y("roas:Q", title="Monthly ROAS"),
        color=alt.Color("campaign_type:N", title="Type"),
        tooltip=["month:T", "campaign_type:N", alt.Tooltip("roas:Q", format=".2f")],
    ).properties(height=280)
    st.altair_chart(trend, width="stretch")

    st.divider()
    st.markdown("**Retention offer lift vs reminder-only (`OFF-NONE`)**")
    lift_path = os.path.join(OUT_DATA, "analytics_offer_lift.csv")
    if os.path.exists(lift_path):
        lift = load_csv(lift_path)
        base = alt.Chart(lift)
        bars = base.mark_bar(color="#2b8a9e").encode(
            x=alt.X("offer_code:N", sort="-y", title=None),
            y=alt.Y("adjusted_lift:Q", title="Adjusted purchase-rate lift"),
            tooltip=["offer_code:N", alt.Tooltip("adjusted_lift:Q", format=".3f"),
                     alt.Tooltip("contacts:Q", format=",.0f")],
        )
        rules = base.mark_rule(color="#333").encode(
            x="offer_code:N", y="lift_ci_low:Q", y2="lift_ci_high:Q",
        )
        st.altair_chart(bars + rules, width="stretch")
        show = lift.copy()
        show.columns = [
            "Offer", "Contacts", "Observed rate", "Baseline rate", "Adjusted lift",
            "CI low", "CI high", "Incremental purchases", "Redemption rate",
            "None rate",
        ]
        for col in ["Observed rate", "Baseline rate", "Adjusted lift", "CI low",
                    "CI high", "Redemption rate", "None rate"]:
            show[col] = show[col].map(lambda x: f"{x:.2%}")
        show["Incremental purchases"] = show["Incremental purchases"].map(lambda x: f"{x:,.1f}")
        show_table(show)
        st.caption(
            "Lift is direct-standardised to the offer group's tier × country × channel mix. "
            "Intervals crossing zero mean the offer effect is not distinguishable from no effect."
        )
    else:
        st.info("Run build_analytics.py to generate offer lift.")

    st.divider()
    st.markdown("**Regional campaign differences-in-differences**")
    did_path = os.path.join(OUT_DATA, "analytics_campaign_did.csv")
    if os.path.exists(did_path):
        did = load_csv(did_path)
        c1, c2 = st.columns(2)
        dcountry = c1.multiselect("DiD countries", sorted(did["country"].dropna().unique()),
                                  default=sorted(did["country"].dropna().unique()))
        dtype = c2.multiselect("DiD types", sorted(did["campaign_type"].dropna().unique()),
                               default=sorted(did["campaign_type"].dropna().unique()))
        view = did[did["country"].isin(dcountry) & did["campaign_type"].isin(dtype)].copy()
        view = view.sort_values("relative_lift", ascending=False)
        view = view[[
            "campaign_id", "campaign_type", "country", "target_region", "start_date",
            "end_date", "post_days", "n_control_regions", "did_daily_revenue",
            "relative_lift", "target_pre_orders", "target_post_orders",
        ]].copy()
        view.columns = [
            "Campaign", "Type", "Country", "Target region", "Start", "End", "Days",
            "Controls", "DiD daily revenue", "Relative lift", "Target pre orders",
            "Target post orders",
        ]
        view["DiD daily revenue"] = view["DiD daily revenue"].map(fmt_money)
        view["Relative lift"] = view["Relative lift"].map(fmt_pct)
        show_table(view.head(100))
    else:
        st.info("Run build_analytics.py to generate DiD results.")


def tab_seasonality() -> None:
    st.subheader("Seasonality: months and years")
    st.caption(
        "Revenue is strongly seasonal: November (Singles' Day / Black Friday) and December are "
        "the busiest months, and dormant customers are more likely to return then. Use this view "
        "so a seasonal lull is not mistaken for churn, and so incentives are not spent on demand "
        "that would have happened anyway."
    )
    month = analytics("analytics_seasonality_month.csv")
    month["month"] = pd.to_datetime(month["month"])
    calendar = analytics("analytics_seasonality_calendar.csv")
    month_order = calendar.sort_values("month_num")["month_name"].tolist()

    countries = sorted(month["country"].dropna().unique())
    selected = st.multiselect("Countries", countries, default=countries,
                              key="seasonality_countries")
    m = month[month["country"].isin(selected)].copy()
    if m.empty:
        st.warning("No rows in the selected countries.")
        return

    holiday = calendar[calendar["is_holiday_season"]]
    nonholiday = calendar[~calendar["is_holiday_season"]]
    peak = calendar.loc[calendar["avg_revenue_index"].idxmax()]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Holiday revenue index", f"{holiday['avg_revenue_index'].mean():.2f}x")
    k2.metric("Peak month", f"{peak['month_name']} ({peak['avg_revenue_index']:.2f}x)")
    k3.metric("Holiday reactivation (180d)",
              fmt_pct(holiday["avg_reactivation_rate_180d"].mean()))
    k4.metric("Non-holiday reactivation",
              fmt_pct(nonholiday["avg_reactivation_rate_180d"].mean()))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Average monthly revenue index vs that year's mean**")
        bar = alt.Chart(calendar).mark_bar().encode(
            x=alt.X("month_name:N", sort=month_order, title=None),
            y=alt.Y("avg_revenue_index:Q", title="Revenue index"),
            color=alt.Color(
                "is_holiday_season:N",
                scale=alt.Scale(domain=[False, True], range=["#9ecae1", "#e6550d"]),
                legend=alt.Legend(title="Holiday season"),
            ),
            tooltip=["month_name:N", alt.Tooltip("avg_revenue_index:Q", format=".2f"),
                     alt.Tooltip("avg_reactivation_rate_180d:Q", format=".3f")],
        ).properties(height=300)
        st.altair_chart(bar, width="stretch")
    with c2:
        st.markdown("**Dormant-customer reactivation rate (180d) by month**")
        line = alt.Chart(calendar).mark_line(point=True).encode(
            x=alt.X("month_name:N", sort=month_order, title=None),
            y=alt.Y("avg_reactivation_rate_180d:Q", title="Reactivation rate",
                    axis=alt.Axis(format="%")),
            tooltip=["month_name:N",
                     alt.Tooltip("avg_reactivation_rate_180d:Q", format=".3%")],
        ).properties(height=300)
        st.altair_chart(line, width="stretch")

    st.markdown("**Monthly revenue index by country and year**")
    heat = alt.Chart(m).mark_rect().encode(
        x=alt.X("year:O", title=None),
        y=alt.Y("month_name:N", sort=month_order, title=None),
        color=alt.Color(
            "revenue_index_vs_year_mean:Q", title="Index vs year mean",
            scale=alt.Scale(scheme="orangered"),
        ),
        tooltip=["country:N", "year:O", "month_name:N",
                 alt.Tooltip("revenue:Q", format="$,.0f"),
                 alt.Tooltip("revenue_index_vs_year_mean:Q", format=".2f"),
                 alt.Tooltip("revenue_yoy_change:Q", format=".1%")],
    ).properties(height=320)
    st.altair_chart(heat, width="stretch")

    st.markdown("**Year-over-year revenue change by month**")
    yoy = m.dropna(subset=["revenue_yoy_change"]).copy()
    if yoy.empty:
        st.info("No year-over-year pairs available for the selected countries.")
    else:
        trend = alt.Chart(yoy).mark_line(point=True).encode(
            x=alt.X("month:T", title=None),
            y=alt.Y("revenue_yoy_change:Q", title="YoY change", axis=alt.Axis(format="%")),
            color=alt.Color("country:N", title="Country"),
            tooltip=["month:T", "country:N",
                     alt.Tooltip("revenue_yoy_change:Q", format=".1%")],
        ).properties(height=280)
        st.altair_chart(trend, width="stretch")

    st.markdown("**Seasonal calendar (all countries)**")
    show = calendar.copy()
    show.columns = [
        "Month #", "Month", "Observations", "Revenue index", "Buyer rate",
        "Reactivation 180d", "Reactivation 365d", "Avg YoY revenue", "Holiday season",
    ]
    show["Revenue index"] = show["Revenue index"].map(lambda x: f"{x:.2f}x")
    for col in ["Buyer rate", "Reactivation 180d", "Reactivation 365d", "Avg YoY revenue"]:
        show[col] = show[col].map(fmt_pct)
    show_table(show)
    st.caption(
        "Revenue index = month revenue divided by that country-year's average month. "
        "Reactivation counts buyers whose previous order was 180+ days earlier, so it shows "
        "previously dormant customers coming back."
    )


def tab_propensity() -> None:
    st.subheader("W1 purchase propensity")
    rolling_metrics_path = os.path.join(ROLLING_REPORTS, "rolling_metrics.json")
    rolling_comparison_path = os.path.join(
        ROLLING_REPORTS, "rolling_model_comparison.csv"
    )
    rolling_importance_path = os.path.join(
        ROLLING_REPORTS, "rolling_feature_importance.csv"
    )
    prep_summary_path = os.path.join(OUT_DATA, "prep_summary.json")
    prep_summary = load_json(prep_summary_path) if os.path.exists(prep_summary_path) else {}

    if os.path.exists(rolling_metrics_path):
        metrics = load_json(rolling_metrics_path)
        comparison = load_csv(rolling_comparison_path)
        importance = load_csv(rolling_importance_path) if os.path.exists(
            rolling_importance_path
        ) else None
        selected_metrics = metrics["validation"]
        positives = int(round(
            float(prep_summary.get("current_w1_high_share", 0.0))
            * int(prep_summary.get("current_customers", 0))
        ))
        caption = (
            f"The rolling W1 model predicts a purchase in the 30 days after the "
            f"{prep_summary.get('score_as_of', metrics.get('test_snapshots', ['latest'])[-1])} "
            f"snapshot (target window "
            f"{prep_summary.get('score_window_start', '?')} to "
            f"{prep_summary.get('score_window_end', '?')}). "
            "Scores are shown at segment level with pseudonymous references only."
        )
    else:
        metrics = load_json(os.path.join(FIXED_REPORTS, "metrics.json"))
        comparison = load_csv(os.path.join(FIXED_REPORTS, "model_comparison.csv"))
        importance = load_csv(
            os.path.join(FIXED_REPORTS, "feature_importance_hgb.csv")
        )
        selected_metrics = next(
            m for m in metrics["models"] if m["model"] == metrics["selected_model"]
        )["at_best_f1"]
        positives = metrics["submission_positives"]
        caption = (
            "The W1 model predicts a purchase in the 30 days after the fixed snapshot. "
            "Scores are shown at segment level with pseudonymous references only."
        )

    scores = analytics("analytics_w1_scores.csv")
    segments = analytics("analytics_w1_segments.csv")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Selected model", metrics["selected_model"])
    k2.metric("Validation ROC-AUC", f"{selected_metrics['roc_auc']:.4f}")
    k3.metric("Validation PR-AUC", f"{selected_metrics['pr_auc']:.4f}")
    k4.metric("Decision threshold", f"{metrics['selected_threshold']:.3f}")
    k5.metric("Submission positives", f"{positives:,}")

    st.caption(caption)
    chart_data = comparison.melt(
        id_vars="model", value_vars=["roc_auc", "pr_auc"],
        var_name="Metric", value_name="Score",
    )
    chart_data["Metric"] = chart_data["Metric"].map({"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"})
    chart = alt.Chart(chart_data).mark_bar().encode(
        y=alt.Y("model:N", sort="-x", title=None),
        x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Metric:N", title=None),
        yOffset="Metric:N",
        tooltip=["model:N", "Metric:N", alt.Tooltip("Score:Q", format=".4f")],
    ).properties(height=260)
    st.altair_chart(chart, width="stretch")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        hist = alt.Chart(scores).mark_bar().encode(
            x=alt.X("w1_probability:Q", bin=alt.Bin(maxbins=35), title="Predicted purchase probability"),
            y=alt.Y("count():Q", title="Customers"),
        ).properties(height=260)
        st.altair_chart(hist, width="stretch")
    with c2:
        seg = segments.copy()
        seg["tier_country"] = seg["cust_tier"] + " · " + seg["cust_country"]
        seg = seg.sort_values("mean_w1_probability", ascending=False).head(12)
        show = seg[["tier_country", "customers", "mean_w1_probability",
                    "high_propensity_share", "expected_buyers"]].copy()
        show.columns = ["Segment", "Customers", "Mean probability", "High-propensity share",
                        "Expected buyers"]
        show["Mean probability"] = show["Mean probability"].map(fmt_pct)
        show["High-propensity share"] = show["High-propensity share"].map(fmt_pct)
        show["Expected buyers"] = show["Expected buyers"].map(lambda x: f"{x:,.0f}")
        show_table(show)

    if importance is not None:
        st.markdown("**Top feature importance**")
        show = importance.head(20).copy()
        show.columns = ["Feature", "Importance"]
        st.dataframe(show, width="stretch", hide_index=True)
    else:
        st.info("Feature importance has not been generated for the rolling model yet.")


def tab_recommendations() -> None:
    st.subheader("W3 weekly retention plan")
    st.caption(
        "Plans use the offer-response model and are constrained by country budget, consent, "
        "service-recovery routing, and a contact cooldown. Costs are redemption-weighted "
        "expected incentive spend."
    )
    budget = read_budget()
    weeks = sorted(budget["week_start"].dt.date.unique())
    countries = sorted(budget["country"].unique())
    c1, c2, c3, c4 = st.columns(4)
    week = c1.selectbox("Week", weeks)
    country = c2.selectbox("Country", countries)
    objective = c3.selectbox(
        "Objective", list(planner.OBJECTIVES.keys()),
        format_func=lambda x: planner.OBJECTIVES[x],
        index=0,
    )
    budget_default = planner.budget_for(pd.Timestamp(week), country)
    budget_usd = c4.number_input("Budget cap (USD)", min_value=0.0, value=float(budget_default),
                                 step=500.0)

    c1, c2, c3, c4 = st.columns(4)
    min_uplift = c1.slider("Minimum uplift", 0.0, 0.20, 0.0, 0.005, format="%.3f")
    min_net = c2.number_input("Minimum expected net per contact", value=0.0, step=1.0)
    cooldown = c3.slider("Contact cooldown (days)", 0, 60, 14, 1)
    max_contacts = c4.number_input("Max contacts (0 = no cap)", min_value=0, value=0, step=1000)

    plan_key = (
        str(week), country, objective, round(float(budget_usd), 2),
        round(float(min_uplift), 4), round(float(min_net), 2),
        int(cooldown), int(max_contacts),
    )
    if st.button("Build plan", type="primary"):
        with st.spinner("Scoring and allocating ..."):
            plan, summary = planner.build_plan(
                week, country, budget_usd, objective, min_uplift,
                min_net_value=min_net,
                max_contacts=int(max_contacts) if max_contacts else None,
                cooldown_days=cooldown,
                scores=load_scores(),
            )
        st.session_state["plan"] = plan
        st.session_state["plan_summary"] = summary
        st.session_state["plan_key"] = plan_key

    plan = st.session_state.get("plan")
    summary = st.session_state.get("plan_summary")
    if plan is None or summary is None:
        st.info("Choose the week, country and guardrails, then build a plan.")
        return
    if st.session_state.get("plan_key") != plan_key:
        st.warning("Settings changed since this plan was built. Build a new plan to continue.")
        return
    st.caption(
        f"Plan for {summary['week_start']} · {summary['country']} · "
        f"{summary['objective_label']} · Budget {fmt_money(summary['budget_usd'])}"
    )
    if summary.get("score_as_of"):
        st.caption(
            f"W1 live snapshot {summary['score_as_of']} · target window "
            f"{summary.get('score_window_start')} to {summary.get('score_window_end')} · "
            f"model {summary.get('score_model') or 'rolling ensemble'}"
        )
    seasonal_note = summary.get("seasonal_note")
    if seasonal_note:
        st.info(f"Seasonality adjustment: {seasonal_note}")
    if plan.empty:
        st.warning(
            "No recommendations meet the current guardrails. Lower the minimum net value, "
            "reduce the cooldown, or choose another objective."
        )
        return

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Contacts", f"{summary['planned_contacts']:,}")
    k2.metric("Expected spend", fmt_money(summary["planned_spend"]))
    k3.metric("Incremental revenue", fmt_money(summary["expected_incremental_revenue"]))
    k4.metric("Expected net margin", fmt_money(summary["expected_net_value"]))
    k5.metric("Expected ROI", f"{summary['expected_roi']:.2f}" if summary["expected_roi"] is not None else "n/a")

    c1, c2 = st.columns(2)
    with c1:
        offer_mix = pd.DataFrame(
            {"offer": list(summary["offer_mix"].keys()),
             "contacts": list(summary["offer_mix"].values())}
        )
        st.altair_chart(
            alt.Chart(offer_mix).mark_bar().encode(
                x=alt.X("contacts:Q", title="Contacts"),
                y=alt.Y("offer:N", sort="-x", title=None),
                tooltip=["offer:N", "contacts:Q"],
            ).properties(height=220),
            width="stretch",
        )
    with c2:
        channel_mix = pd.DataFrame(
            {"channel": list(summary["channel_mix"].keys()),
             "contacts": list(summary["channel_mix"].values())}
        )
        st.altair_chart(
            alt.Chart(channel_mix).mark_bar().encode(
                x=alt.X("contacts:Q", title="Contacts"),
                y=alt.Y("channel:N", sort="-x", title=None),
                color=alt.Color("channel:N", legend=None),
                tooltip=["channel:N", "contacts:Q"],
            ).properties(height=220),
            width="stretch",
        )

    display_cols = [
        "priority_rank", "sample_ref", "cust_country", "cust_tier", "channel", "offer_code",
        "purchase_probability", "baseline_probability", "uplift", "redemption_probability",
        "offer_cost", "expected_incremental_revenue", "expected_net_value", "roi",
        "w1_probability", "priority_score",
    ]
    show = plan[[c for c in display_cols if c in plan.columns]].copy()
    show.columns = [
        "Rank", "Sample ref", "Country", "Tier", "Channel", "Offer",
        "Purchase prob.", "Baseline prob.", "Uplift", "Redemption prob.",
        "Expected cost", "Incremental revenue", "Expected net", "ROI",
        "W1 prob.", "Priority score",
    ][:len(show.columns)]
    for col in ["Purchase prob.", "Baseline prob.", "Uplift", "Redemption prob.", "W1 prob."]:
        if col in show.columns:
            show[col] = show[col].map(fmt_pct)
    for col in ["Expected cost", "Incremental revenue", "Expected net", "Priority score"]:
        if col in show.columns:
            show[col] = show[col].map(fmt_money)
    if "ROI" in show.columns:
        show["ROI"] = show["ROI"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a")
    show_table(show.head(500))
    st.caption(f"Showing up to 500 of {len(plan):,} recommendations. The CSV contains all rows.")

    csv = plan.to_csv(index=False).encode("utf-8")
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.download_button(
        "Download plan CSV", csv,
        file_name=f"plan_{week}_{country}.csv", mime="text/csv",
        width="stretch",
    )
    with c2:
        prepared_by = st.text_input("Prepared by", value="", key="prepared_by")
    with c3:
        st.write("")
        st.write("")
        created_ref = (
            st.session_state.get("batch_ref")
            if st.session_state.get("batch_key") == plan_key else None
        )
        if created_ref:
            st.info(f"Approval batch {created_ref} already created for this plan.")
        elif st.button("Create approval batch", width="stretch"):
            try:
                ref = approvals.create_batch(plan, summary, prepared_by or "planner")
                st.session_state["batch_key"] = plan_key
                st.session_state["batch_ref"] = ref
                st.success(f"Created {ref}. Open the Approvals tab to review it.")
            except Exception as exc:
                st.error(str(exc))


def tab_approvals() -> None:
    st.subheader("W9 human approval and audit")
    batches = approvals.list_batches()
    if batches.empty:
        st.info("No approval batches yet. Build a W3 plan and create a batch first.")
        return
    show = batches[[
        "batch_ref", "created_at", "week_start", "country", "objective", "planned_contacts",
        "planned_spend", "expected_net_value", "status", "decided_by",
    ]].copy()
    show.columns = [
        "Batch", "Created", "Week", "Country", "Objective", "Contacts",
        "Expected spend", "Expected net", "Status", "Decided by",
    ]
    show["Expected spend"] = show["Expected spend"].map(fmt_money)
    show["Expected net"] = show["Expected net"].map(fmt_money)
    show_table(show)

    refs = batches["batch_ref"].tolist()
    selected = st.selectbox("Open batch", refs)
    batch, recs = approvals.get_batch(selected)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Status", batch["status"])
    k2.metric("Contacts", f"{int(batch['planned_contacts']):,}")
    k3.metric("Expected spend", fmt_money(batch["planned_spend"]))
    k4.metric("Expected net", fmt_money(batch["expected_net_value"]))

    display = recs[[
        "priority_rank", "sample_ref", "cust_tier", "channel", "offer_code", "uplift",
        "expected_order_value", "offer_cost", "expected_net_value", "roi",
    ]].copy()
    display.columns = [
        "Rank", "Sample ref", "Tier", "Channel", "Offer", "Uplift",
        "Expected order value", "Expected cost", "Expected net", "ROI",
    ]
    display["Uplift"] = display["Uplift"].map(fmt_pct)
    for col in ["Expected order value", "Expected cost", "Expected net"]:
        display[col] = display[col].map(fmt_money)
    display["ROI"] = display["ROI"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "n/a")
    show_table(display.head(500))

    st.markdown("**Audit trail**")
    trail = approvals.audit_trail(selected)
    if trail.empty:
        st.caption("No audit events.")
    else:
        trail = trail[["created_at", "action", "actor", "reason"]].copy()
        trail.columns = ["Time", "Action", "Actor", "Reason"]
        show_table(trail)

    if batch["status"] == "Pending":
        with st.form("approval_form"):
            decision = st.radio("Decision", ["Approved", "Rejected"], horizontal=True)
            actor = st.text_input("Reviewer name")
            reason = st.text_area("Reason for decision")
            submitted = st.form_submit_button("Record decision", type="primary")
        if submitted:
            try:
                approvals.decide_batch(selected, decision, actor, reason)
                st.success(f"{selected} marked {decision}.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    else:
        st.info(
            f"Decided by {batch['decided_by']} on {batch['decided_at']}: "
            f"{batch['decision_reason']}"
        )

    if batch["status"] == "Approved":
        if st.button("Export approved CSV"):
            try:
                path = approvals.export_approved(selected)
                with open(path, "rb") as fh:
                    st.download_button(
                        "Download approved batch", fh.read(),
                        file_name=os.path.basename(path), mime="text/csv",
                        width="stretch",
                    )
                st.success(f"Exported {os.path.basename(path)}")
            except Exception as exc:
                st.error(str(exc))


def main() -> None:
    st.title("ShopSphere retention intelligence")
    st.caption(
        "W1 propensity · W3 budgeted recommendations · W6 performance analytics · "
        "W9 approvals · W10 effectiveness"
    )
    tabs = st.tabs([
        "Overview (W6)", "Campaigns (W10)", "Seasonality (W6)", "Propensity (W1)",
        "Recommendations (W3)", "Approvals (W9)",
    ])
    with tabs[0]:
        tab_overview()
    with tabs[1]:
        tab_campaigns()
    with tabs[2]:
        tab_seasonality()
    with tabs[3]:
        tab_propensity()
    with tabs[4]:
        tab_recommendations()
    with tabs[5]:
        tab_approvals()


if __name__ == "__main__":
    main()
