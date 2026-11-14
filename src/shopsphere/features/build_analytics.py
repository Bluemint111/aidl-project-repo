from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from shopsphere.common import (
    ANALYTICS_REPORTS, OUT_DATA, ensure_dirs, read_csv, region_country_map, to01,
)
from shopsphere.data.asof_time import parse_datetime

STRATA = ["cust_tier", "country", "channel"]


def customer_map() -> pd.DataFrame:
    c = read_csv("customers.csv", usecols=["customer_id", "region", "country_code_clean"])
    return c.rename(columns={"region": "cust_region", "country_code_clean": "country"})


def daily_country() -> pd.DataFrame:
    cust = customer_map()
    tx = read_csv(
        "transactions.csv",
        usecols=["transaction_id", "customer_id", "transaction_timestamp", "order_value",
                 "items_count"],
    )
    tx["date"] = parse_datetime(tx["transaction_timestamp"]).dt.normalize()
    tx = tx.merge(cust[["customer_id", "country", "cust_region"]], on="customer_id", how="left")

    sess = read_csv("shopsphere_sessions_cleaned.csv", usecols=["customer_id", "session_timestamp"])
    sess["date"] = parse_datetime(sess["session_timestamp"]).dt.normalize()
    sess = sess.merge(cust[["customer_id", "country", "cust_region"]], on="customer_id", how="left")

    camp = read_csv(
        "marketing_campaigns.csv",
        usecols=["campaign_id", "campaign_type", "region_target", "start_date", "end_date",
                 "status", "campaign_budget"],
    )
    rmap = region_country_map()
    camp["country"] = camp["region_target"].map(rmap)
    perf = read_csv("campaign_daily_performance.csv")
    perf["date"] = parse_datetime(perf["date"]).dt.normalize()
    perf = perf.merge(
        camp[["campaign_id", "campaign_type", "region_target", "country"]],
        on="campaign_id", how="left",
    )

    d = read_csv(
        "deliveries.csv",
        usecols=["transaction_id", "carrier", "ship_date", "delivered_date", "delay_days",
                 "delivery_status"],
    )
    d["date"] = parse_datetime(d["delivered_date"]).fillna(parse_datetime(d["ship_date"]))
    d["date"] = d["date"].dt.normalize()
    d = d.merge(tx[["transaction_id", "country", "cust_region"]], on="transaction_id", how="left")
    d["_late"] = (d["delivery_status"] == "Delivered Late").astype(int)
    d["_lost"] = (d["delivery_status"] == "Lost").astype(int)
    d["_returned"] = (d["delivery_status"] == "Returned").astype(int)

    tx_daily = tx.groupby(["date", "country"], dropna=False).agg(
        revenue=("order_value", "sum"),
        orders=("transaction_id", "nunique"),
        items=("items_count", "sum"),
    ).reset_index()
    sess_daily = sess.groupby(["date", "country"], dropna=False).agg(
        sessions=("customer_id", "size"),
    ).reset_index()
    perf_daily = perf.groupby(["date", "country"], dropna=False).agg(
        campaign_spend=("spend", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        attributed_sessions=("attributed_sessions", "sum"),
        attributed_orders=("attributed_orders", "sum"),
        attributed_revenue=("attributed_revenue", "sum"),
    ).reset_index()
    del_daily = d.groupby(["date", "country"], dropna=False).agg(
        deliveries=("transaction_id", "size"),
        late_deliveries=("_late", "sum"),
        lost_deliveries=("_lost", "sum"),
        returned_deliveries=("_returned", "sum"),
    ).reset_index()

    out = tx_daily.merge(sess_daily, on=["date", "country"], how="outer")
    out = out.merge(perf_daily, on=["date", "country"], how="outer")
    out = out.merge(del_daily, on=["date", "country"], how="outer")
    num_cols = ["revenue", "orders", "items", "sessions", "campaign_spend", "impressions",
                "clicks", "attributed_sessions", "attributed_orders", "attributed_revenue",
                "deliveries", "late_deliveries", "lost_deliveries", "returned_deliveries"]
    for col in num_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["date", "country"]).sort_values(["date", "country"])
    out["aov"] = out["revenue"] / out["orders"].replace(0, np.nan)
    out["conversion_rate"] = out["orders"] / out["sessions"].replace(0, np.nan)
    out["campaign_roas"] = out["attributed_revenue"] / out["campaign_spend"].replace(0, np.nan)
    out["late_rate"] = out["late_deliveries"] / out["deliveries"].replace(0, np.nan)
    out.to_csv(os.path.join(OUT_DATA, "analytics_daily_country.csv"), index=False)
    return out


def country_scorecard(daily: pd.DataFrame) -> pd.DataFrame:
    end = pd.Timestamp(daily["date"].max())
    last_start = end - pd.Timedelta(days=89)
    prior_end = last_start - pd.Timedelta(days=1)
    prior_start = prior_end - pd.Timedelta(days=89)

    def agg(start, end_date, suffix):
        d = daily[(daily["date"] >= start) & (daily["date"] <= end_date)]
        g = d.groupby("country").agg(
            revenue=("revenue", "sum"), orders=("orders", "sum"), items=("items", "sum"),
            sessions=("sessions", "sum"), campaign_spend=("campaign_spend", "sum"),
            attributed_revenue=("attributed_revenue", "sum"),
            deliveries=("deliveries", "sum"), late=("late_deliveries", "sum"),
        )
        g["aov"] = g["revenue"] / g["orders"].replace(0, np.nan)
        g["conversion"] = g["orders"] / g["sessions"].replace(0, np.nan)
        g["roas"] = g["attributed_revenue"] / g["campaign_spend"].replace(0, np.nan)
        g["late_rate"] = g["late"] / g["deliveries"].replace(0, np.nan)
        return g.add_suffix(suffix)

    out = agg(last_start, end, "_last90").join(agg(prior_start, prior_end, "_prior90"), how="outer")
    out["revenue_change_pct"] = (
        out["revenue_last90"] / out["revenue_prior90"].replace(0, np.nan) - 1
    )
    out["orders_change_pct"] = (
        out["orders_last90"] / out["orders_prior90"].replace(0, np.nan) - 1
    )
    out["conversion_change"] = out["conversion_last90"] - out["conversion_prior90"]
    out["aov_change_pct"] = out["aov_last90"] / out["aov_prior90"].replace(0, np.nan) - 1
    out = out.reset_index()
    out["period_end"] = end.date().isoformat()
    out.to_csv(os.path.join(OUT_DATA, "analytics_country_scorecard.csv"), index=False)
    return out


def campaign_analytics() -> tuple[pd.DataFrame, pd.DataFrame]:
    camp = read_csv(
        "marketing_campaigns.csv",
        usecols=["campaign_id", "campaign_type", "region_target", "start_date", "end_date",
                 "status", "campaign_budget"],
    )
    rmap = region_country_map()
    camp["country"] = camp["region_target"].map(rmap)
    perf = read_csv("campaign_daily_performance.csv")
    perf["date"] = parse_datetime(perf["date"]).dt.normalize()
    perf["month"] = perf["date"].dt.to_period("M").dt.to_timestamp()
    perf = perf.merge(
        camp[["campaign_id", "campaign_type", "region_target", "country"]],
        on="campaign_id", how="left",
    )
    monthly = perf.groupby(["month", "campaign_type"], dropna=False).agg(
        campaigns=("campaign_id", "nunique"),
        spend=("spend", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        attributed_sessions=("attributed_sessions", "sum"),
        attributed_orders=("attributed_orders", "sum"),
        attributed_revenue=("attributed_revenue", "sum"),
    ).reset_index()
    monthly["ctr"] = monthly["clicks"] / monthly["impressions"].replace(0, np.nan)
    monthly["cpa"] = monthly["spend"] / monthly["attributed_orders"].replace(0, np.nan)
    monthly["roas"] = monthly["attributed_revenue"] / monthly["spend"].replace(0, np.nan)
    monthly.to_csv(os.path.join(OUT_DATA, "analytics_campaign_type.csv"), index=False)

    by_country = perf.groupby(["country", "campaign_type"], dropna=False).agg(
        campaigns=("campaign_id", "nunique"),
        spend=("spend", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        attributed_orders=("attributed_orders", "sum"),
        attributed_revenue=("attributed_revenue", "sum"),
    ).reset_index()
    by_country["ctr"] = by_country["clicks"] / by_country["impressions"].replace(0, np.nan)
    by_country["cpa"] = by_country["spend"] / by_country["attributed_orders"].replace(0, np.nan)
    by_country["roas"] = by_country["attributed_revenue"] / by_country["spend"].replace(0, np.nan)
    by_country.to_csv(os.path.join(OUT_DATA, "analytics_campaign_country.csv"), index=False)
    return monthly, by_country


def campaign_did() -> pd.DataFrame:
    cust = customer_map()
    tx = read_csv(
        "transactions.csv",
        usecols=["transaction_id", "customer_id", "transaction_timestamp", "order_value"],
    )
    tx["date"] = parse_datetime(tx["transaction_timestamp"]).dt.normalize()
    tx = tx.merge(cust[["customer_id", "cust_region", "country"]], on="customer_id", how="left")
    reg_daily = tx.groupby(["date", "cust_region"], dropna=False).agg(
        revenue=("order_value", "sum"), orders=("transaction_id", "nunique"),
    ).reset_index()
    all_regions = sorted(cust["cust_region"].dropna().unique())
    revenue = reg_daily.pivot_table(index="cust_region", columns="date", values="revenue",
                                    fill_value=0.0, aggfunc="sum").reindex(all_regions, fill_value=0.0)
    orders = reg_daily.pivot_table(index="cust_region", columns="date", values="orders",
                                   fill_value=0.0, aggfunc="sum").reindex(all_regions, fill_value=0.0)

    rmap = region_country_map()
    camp = read_csv(
        "marketing_campaigns.csv",
        usecols=["campaign_id", "campaign_type", "region_target", "start_date", "end_date",
                 "status", "campaign_budget"],
    )
    camp["start_date"] = parse_datetime(camp["start_date"])
    camp["end_date"] = parse_datetime(camp["end_date"])
    camp["country"] = camp["region_target"].map(rmap)
    regions_by_country = {
        c: sorted([r for r in all_regions if rmap.get(r) == c])
        for c in camp["country"].dropna().unique()
    }

    rows = []
    for _, c in camp.iterrows():
        if pd.isna(c["start_date"]) or pd.isna(c["end_date"]) or pd.isna(c["country"]):
            continue
        start, end = c["start_date"], c["end_date"]
        post_days = (end - start).days + 1
        if post_days <= 0:
            continue
        pre_end = start - pd.Timedelta(days=1)
        pre_start = pre_end - pd.Timedelta(days=post_days - 1)
        active = camp[(camp["start_date"] <= end) & (camp["end_date"] >= start)]
        blocked = set(active["region_target"].dropna())
        control = [r for r in regions_by_country.get(c["country"], [])
                   if r not in blocked and r != c["region_target"]]
        if len(control) < 3:
            continue
        target = c["region_target"]
        if target not in revenue.index:
            continue
        tpre = revenue.loc[target, pre_start:pre_end].sum()
        tpost = revenue.loc[target, start:end].sum()
        cpre = revenue.loc[control, pre_start:pre_end].sum().sum()
        cpost = revenue.loc[control, start:end].sum().sum()
        tpre_o = orders.loc[target, pre_start:pre_end].sum()
        tpost_o = orders.loc[target, start:end].sum()
        cpre_o = orders.loc[control, pre_start:pre_end].sum().sum()
        cpost_o = orders.loc[control, start:end].sum().sum()
        if tpre_o < 5 or tpost_o < 5:
            continue
        tpre_d = tpre / post_days
        tpost_d = tpost / post_days
        cpre_d = cpre / (post_days * len(control))
        cpost_d = cpost / (post_days * len(control))
        did = (tpost_d - tpre_d) - (cpost_d - cpre_d)
        rows.append({
            "campaign_id": int(c["campaign_id"]),
            "campaign_type": c["campaign_type"],
            "country": c["country"],
            "target_region": target,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "post_days": int(post_days),
            "n_control_regions": len(control),
            "target_pre_daily_revenue": round(float(tpre_d), 2),
            "target_post_daily_revenue": round(float(tpost_d), 2),
            "control_pre_daily_revenue": round(float(cpre_d), 2),
            "control_post_daily_revenue": round(float(cpost_d), 2),
            "did_daily_revenue": round(float(did), 2),
            "relative_lift": round(float(did / tpre_d), 5) if tpre_d else None,
            "target_pre_orders": int(tpre_o),
            "target_post_orders": int(tpost_o),
            "control_pre_orders": int(cpre_o),
            "control_post_orders": int(cpost_o),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("relative_lift", ascending=False)
    out.to_csv(os.path.join(OUT_DATA, "analytics_campaign_did.csv"), index=False)
    return out


def carrier_analytics() -> pd.DataFrame:
    cust = customer_map()
    tx = read_csv("transactions.csv", usecols=["transaction_id", "customer_id"])
    tx = tx.merge(cust[["customer_id", "country"]], on="customer_id", how="left")
    d = read_csv(
        "deliveries.csv",
        usecols=["transaction_id", "carrier", "delay_days", "delivery_status"],
    )
    d = d.merge(tx, on="transaction_id", how="left")
    d["_late"] = (d["delivery_status"] == "Delivered Late").astype(int)
    d["_lost"] = (d["delivery_status"] == "Lost").astype(int)
    d["_returned"] = (d["delivery_status"] == "Returned").astype(int)
    g = d.groupby(["country", "carrier"], dropna=False).agg(
        deliveries=("transaction_id", "size"),
        late=("_late", "sum"),
        lost=("_lost", "sum"),
        returned=("_returned", "sum"),
        avg_delay_days=("delay_days", "mean"),
    ).reset_index()
    g["late_rate"] = g["late"] / g["deliveries"].replace(0, np.nan)
    g["lost_rate"] = g["lost"] / g["deliveries"].replace(0, np.nan)
    g["returned_rate"] = g["returned"] / g["deliveries"].replace(0, np.nan)
    g.to_csv(os.path.join(OUT_DATA, "analytics_carrier.csv"), index=False)
    return g


def support_analytics() -> pd.DataFrame:
    cust = customer_map()
    s = read_csv(
        "support_tickets.csv",
        usecols=["ticket_id", "customer_id", "created_date", "priority", "status",
                 "first_response_hours", "resolution_hours", "csat_score"],
    )
    s["month"] = parse_datetime(s["created_date"]).dt.to_period("M").dt.to_timestamp()
    s = s.merge(cust[["customer_id", "country"]], on="customer_id", how="left")
    s["_open"] = (s["status"] == "Open").astype(int)
    s["_high"] = (s["priority"] == "High").astype(int)
    s["_low_csat"] = (s["csat_score"] <= 2).astype(int)
    g = s.groupby(["month", "country"], dropna=False).agg(
        tickets=("ticket_id", "size"),
        open_tickets=("_open", "sum"),
        high_priority=("_high", "sum"),
        low_csat=("_low_csat", "sum"),
        csat_mean=("csat_score", "mean"),
        first_response_mean=("first_response_hours", "mean"),
        resolution_mean=("resolution_hours", "mean"),
    ).reset_index()
    g.to_csv(os.path.join(OUT_DATA, "analytics_support_country.csv"), index=False)
    return g


def standardized_lift(offer_df: pd.DataFrame, none_df: pd.DataFrame,
                      strata: list[str]) -> tuple[float, float, float, float]:
    overall = float(none_df["purchase"].mean())
    observed = float(offer_df["purchase"].mean())
    weights = offer_df.groupby(strata).size().div(len(offer_df))
    rates = none_df.groupby(strata)["purchase"].mean()
    expected = float((weights * rates.reindex(weights.index).fillna(overall)).sum())
    return observed, expected, observed - expected, overall


def offer_lift() -> pd.DataFrame:
    r = read_csv("retention_contact_history.csv")
    r["purchase"] = to01(r["purchased_within_14d_clean"]).fillna(0).astype(int)
    r["redeemed"] = to01(r["redeemed_clean"])
    c = read_csv("customers.csv", usecols=["customer_id", "customer_tier", "country_code_clean"])
    r = r.merge(c.rename(columns={"customer_tier": "cust_tier",
                                   "country_code_clean": "country"}),
                on="customer_id", how="left")
    r["channel"] = r["channel"].astype(str)
    none = r[r["offer_code"] == "OFF-NONE"].copy()
    rng = np.random.default_rng(42)
    rows = []
    for offer in sorted([o for o in r["offer_code"].unique() if o != "OFF-NONE"]):
        sub = r[r["offer_code"] == offer].copy()
        if len(sub) < 20:
            continue
        observed, expected, lift, overall_none = standardized_lift(sub, none, STRATA)
        boots = []
        for _ in range(300):
            b_offer = sub.iloc[rng.integers(0, len(sub), len(sub))]
            b_none = none.iloc[rng.integers(0, len(none), len(none))]
            _, _, b_lift, _ = standardized_lift(b_offer, b_none, STRATA)
            boots.append(b_lift)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({
            "offer_code": offer,
            "contacts": int(len(sub)),
            "observed_purchase_rate": round(observed, 5),
            "expected_baseline_rate": round(expected, 5),
            "adjusted_lift": round(lift, 5),
            "lift_ci_low": round(float(lo), 5),
            "lift_ci_high": round(float(hi), 5),
            "incremental_purchases": round(lift * len(sub), 2),
            "redemption_rate": round(float(sub["redeemed"].mean()), 5),
            "overall_none_rate": round(overall_none, 5),
        })
    out = pd.DataFrame(rows).sort_values("adjusted_lift", ascending=False)
    out.to_csv(os.path.join(OUT_DATA, "analytics_offer_lift.csv"), index=False)
    return out


def seasonality_analytics() -> tuple[pd.DataFrame, pd.DataFrame]:
    cust = customer_map()
    tx = read_csv(
        "transactions.csv",
        usecols=["transaction_id", "customer_id", "transaction_timestamp", "order_value"],
    )
    tx["date"] = parse_datetime(tx["transaction_timestamp"]).dt.normalize()
    tx["month"] = tx["date"].dt.to_period("M").dt.to_timestamp()
    tx = tx.merge(cust[["customer_id", "country"]], on="customer_id", how="left")

    sess = read_csv(
        "shopsphere_sessions_cleaned.csv", usecols=["customer_id", "session_timestamp"],
    )
    sess["month"] = parse_datetime(sess["session_timestamp"]).dt.to_period("M").dt.to_timestamp()
    sess = sess.merge(cust[["customer_id", "country"]], on="customer_id", how="left")
    active = sess.groupby(["month", "country"], dropna=False).agg(
        active_customers=("customer_id", "nunique"),
    ).reset_index()

    cm = tx.groupby(["customer_id", "country", "month"], dropna=False).agg(
        revenue=("order_value", "sum"),
        orders=("transaction_id", "nunique"),
        last_order=("date", "max"),
    ).reset_index().sort_values(["customer_id", "month"])
    cm["previous_order"] = cm.groupby("customer_id")["last_order"].shift()
    cm["days_since_previous_order"] = (
        cm["month"] - cm["previous_order"]
    ).dt.days
    cm["first_time_buyer"] = cm["previous_order"].isna()
    cm["reactivated_180d"] = cm["days_since_previous_order"].ge(180)
    cm["reactivated_365d"] = cm["days_since_previous_order"].ge(365)

    out = cm.groupby(["month", "country"], dropna=False).agg(
        revenue=("revenue", "sum"),
        orders=("orders", "sum"),
        buyers=("customer_id", "nunique"),
        first_time_buyers=("first_time_buyer", "sum"),
        reactivated_180d=("reactivated_180d", "sum"),
        reactivated_365d=("reactivated_365d", "sum"),
    ).reset_index()
    out = out.merge(active, on=["month", "country"], how="left")
    out["active_customers"] = out["active_customers"].fillna(0)
    out["buyer_rate"] = out["buyers"] / out["active_customers"].replace(0, np.nan)
    out["reactivation_rate_180d"] = (
        out["reactivated_180d"] / out["buyers"].replace(0, np.nan)
    )
    out["reactivation_rate_365d"] = (
        out["reactivated_365d"] / out["buyers"].replace(0, np.nan)
    )
    out["aov"] = out["revenue"] / out["orders"].replace(0, np.nan)

    prior = out[["country", "month", "revenue", "orders", "buyers", "reactivated_180d"]].copy()
    prior["month"] = prior["month"] + pd.DateOffset(years=1)
    prior = prior.rename(columns={
        "revenue": "revenue_prior_year",
        "orders": "orders_prior_year",
        "buyers": "buyers_prior_year",
        "reactivated_180d": "reactivated_180d_prior_year",
    })
    out = out.merge(prior, on=["country", "month"], how="left")
    for col in ["revenue", "orders", "buyers", "reactivated_180d"]:
        out[f"{col}_yoy_change"] = (
            out[col] / out[f"{col}_prior_year"].replace(0, np.nan) - 1
        )

    out["year"] = out["month"].dt.year
    out["month_num"] = out["month"].dt.month
    out["month_name"] = out["month"].dt.strftime("%b")
    out["is_holiday_season"] = out["month_num"].isin([11, 12])
    out["year_mean_revenue"] = out.groupby(["country", "year"])["revenue"].transform("mean")
    out["revenue_index_vs_year_mean"] = (
        out["revenue"] / out["year_mean_revenue"].replace(0, np.nan)
    )
    out = out.sort_values(["country", "month"]).reset_index(drop=True)
    out.to_csv(os.path.join(OUT_DATA, "analytics_seasonality_month.csv"), index=False)

    calendar = out.groupby(["month_num", "month_name"], dropna=False).agg(
        observations=("month", "size"),
        avg_revenue_index=("revenue_index_vs_year_mean", "mean"),
        avg_buyer_rate=("buyer_rate", "mean"),
        avg_reactivation_rate_180d=("reactivation_rate_180d", "mean"),
        avg_reactivation_rate_365d=("reactivation_rate_365d", "mean"),
        avg_yoy_revenue_change=("revenue_yoy_change", "mean"),
    ).reset_index().sort_values("month_num")
    calendar["is_holiday_season"] = calendar["month_num"].isin([11, 12])
    calendar.to_csv(os.path.join(OUT_DATA, "analytics_seasonality_calendar.csv"), index=False)
    return out, calendar


def w1_segments() -> tuple[pd.DataFrame, pd.DataFrame]:
    cur = pd.read_pickle(os.path.join(OUT_DATA, "current_customers.pkl"))
    keep = ["sample_ref", "cust_country", "cust_tier", "pref_channel", "w1_probability",
            "w1_target", "expected_order_value", "last_contact_date"]
    scores = cur[[c for c in keep if c in cur.columns]].copy()
    scores.to_csv(os.path.join(OUT_DATA, "analytics_w1_scores.csv"), index=False)
    g = scores.groupby(["cust_tier", "cust_country"], dropna=False).agg(
        customers=("sample_ref", "size"),
        mean_w1_probability=("w1_probability", "mean"),
        high_propensity_share=("w1_target", "mean"),
        expected_buyers=("w1_probability", "sum"),
        mean_expected_order_value=("expected_order_value", "mean"),
    ).reset_index()
    g.to_csv(os.path.join(OUT_DATA, "analytics_w1_segments.csv"), index=False)
    return scores, g


def main() -> None:
    ensure_dirs()
    print("daily country")
    daily = daily_country()
    print("country scorecard")
    scorecard = country_scorecard(daily)
    print("campaign analytics")
    monthly, by_country = campaign_analytics()
    print("campaign DiD")
    did = campaign_did()
    print("carrier")
    carrier = carrier_analytics()
    print("support")
    support = support_analytics()
    print("offer lift")
    lift = offer_lift()
    print("seasonality")
    season_month, season_calendar = seasonality_analytics()
    print("W1 segments")
    w1_scores, w1_seg = w1_segments()

    summary = {
        "daily_rows": int(len(daily)),
        "daily_date_min": str(daily["date"].min().date()),
        "daily_date_max": str(daily["date"].max().date()),
        "countries": sorted(daily["country"].dropna().unique().tolist()),
        "scorecard_rows": int(len(scorecard)),
        "campaign_monthly_rows": int(len(monthly)),
        "campaign_country_rows": int(len(by_country)),
        "campaign_did_rows": int(len(did)),
        "carrier_rows": int(len(carrier)),
        "support_rows": int(len(support)),
        "offer_lift_rows": int(len(lift)),
        "seasonality_month_rows": int(len(season_month)),
        "seasonality_calendar_rows": int(len(season_calendar)),
        "holiday_avg_revenue_index": round(
            float(season_calendar.loc[season_calendar["is_holiday_season"],
                                      "avg_revenue_index"].mean()), 4
        ),
        "w1_score_rows": int(len(w1_scores)),
        "w1_segment_rows": int(len(w1_seg)),
    }
    with open(os.path.join(ANALYTICS_REPORTS, "analytics_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
