"""As-of customer feature engineering for rolling ShopSphere predictions.

The original ``build_features.py`` is tied to the October 2024 checkpoint.
This module generalises the same feature blocks so a snapshot can be built for
any as-of date.  It deliberately keeps the original feature names and adds a
target-window seasonality block, so the existing offer model can continue to
use the customer features it was trained on.

Typical use::

    raw = load_raw()
    feats = build_feature_frame(raw, as_of="YYYY-MM-DD")
    labelled = build_snapshot(raw, "YYYY-MM-DD")   # features + target
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from shopsphere.data.asof_time import (
    month_end_dates,
    next_black_friday,
    next_christmas,
    parse_datetime,
    resolve_as_of,
    target_months,
    weekly_dates,
    window_bounds,
)

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(BASE)))
DATA = os.environ.get("SHOPSPHERE_DATA_DIR", os.path.join(PROJECT, "data", "clean"))
WINDOW_DAYS = 30

TRAFFIC = ["organic", "direct", "ads", "email", "social"]
DEVICE = ["mobile", "desktop", "tablet"]
CAMPAIGN_TYPES = ["Email", "Search", "Display", "Social", "Affiliate"]
PAYMENTS = ["credit_card", "debit_card", "paypal", "crypto"]
SHIP_SPEEDS = ["standard", "express", "overnight"]
DELIVERY_STATUS = ["Delivered", "Delivered Late", "Lost", "Returned", "In Transit"]
TICKET_CATS = [
    "Late delivery", "Damaged or wrong item", "Product question", "Refund status",
    "Lost parcel", "Payment issue", "Account access", "Promo code not working", "Cancel order",
]
RET_CHANNELS = ["email", "app_push", "sms"]


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def read(name: str, data_dir: str = DATA, **kwargs) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, name), low_memory=False, **kwargs)


def load_raw(data_dir: str = DATA) -> dict[str, pd.DataFrame]:
    """Load the source tables once; every snapshot reuses the same frames."""
    return {
        "customers": read("customers.csv", data_dir),
        "geo": read("geo_data.csv", data_dir),
        "preferences": read("customer_preferences.csv", data_dir),
        "sessions": read("shopsphere_sessions_cleaned.csv", data_dir),
        "transactions": read("transactions.csv", data_dir),
        "transaction_items": read("transaction_items.csv", data_dir),
        "products": read("products.csv", data_dir),
        "deliveries": read("deliveries.csv", data_dir),
        "support_tickets": read("support_tickets.csv", data_dir),
        "retention_contacts": read("retention_contact_history.csv", data_dir),
        "campaigns": read("marketing_campaigns.csv", data_dir),
        "campaign_performance": read("campaign_daily_performance.csv", data_dir),
        "budget": read("retention_budget_2024Q4.csv", data_dir),
    }


def to01(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("int64")
    low = series.astype("string").str.strip().str.lower()
    return low.map({
        "true": 1, "false": 0, "y": 1, "n": 0, "yes": 1, "no": 0,
        "1": 1, "0": 0, "t": 1, "f": 0,
    })


def days_since(dates: pd.Series, as_of) -> pd.Series:
    return (pd.Timestamp(as_of) - parse_datetime(dates)).dt.days


def shares(df: pd.DataFrame, id_col: str, cat_col: str, categories: list[str], prefix: str) -> pd.DataFrame:
    tmp = df[[id_col, cat_col]].dropna()
    tab = tmp.groupby([id_col, cat_col]).size().unstack(fill_value=0)
    tab = tab.reindex(columns=categories, fill_value=0)
    out = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0)
    out.columns = [f"{prefix}{c}" for c in out.columns]
    return out


def counts(df: pd.DataFrame, id_col: str, cat_col: str, categories: list[str], prefix: str) -> pd.DataFrame:
    tmp = df[[id_col, cat_col]].dropna()
    tab = tmp.groupby([id_col, cat_col]).size().unstack(fill_value=0)
    tab = tab.reindex(columns=categories, fill_value=0)
    tab.columns = [f"{prefix}{c}" for c in tab.columns]
    return tab


# --------------------------------------------------------------------------- #
# Feature blocks
# --------------------------------------------------------------------------- #
def customer_base(raw: dict[str, pd.DataFrame], as_of) -> pd.DataFrame:
    c = raw["customers"].copy()
    c["tenure_days"] = days_since(c["signup_date"], as_of)
    c["email_open_rate_missing"] = c["email_open_rate"].isna().astype(int)
    c["has_referral"] = c["referral_code"].notna().astype(int)
    c["tier_ord"] = c["customer_tier"].map({"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3})
    c = c.rename(columns={
        "age": "cust_age", "country_code_clean": "cust_country", "region": "cust_region",
        "loyalty_score": "cust_loyalty", "email_open_rate": "cust_email_open",
        "discount_usage_rate": "cust_discount_rate", "avg_review_score": "cust_review",
        "customer_tier": "cust_tier", "credit_limit": "cust_credit_limit",
        "email_open_rate_missing": "cust_email_open_missing",
    })
    keep = ["customer_id", "cust_age", "cust_country", "cust_region", "tenure_days",
            "cust_loyalty", "cust_email_open", "cust_email_open_missing",
            "cust_discount_rate", "cust_review", "has_referral", "cust_tier", "tier_ord",
            "cust_credit_limit"]
    base = c[keep].copy()

    geo = raw["geo"].rename(columns={
        "geo_ip_region": "cust_region", "average_income": "geo_income",
        "urban_ratio": "geo_urban", "internet_penetration": "geo_internet",
        "region_tier": "geo_tier",
    })
    base = base.merge(geo, on="cust_region", how="left")

    pref = raw["preferences"].copy()
    pref["pref_days_since_update"] = days_since(pref["last_updated"], as_of)
    for col in ["email_opt_in_clean", "sms_opt_in_clean", "do_not_contact_clean"]:
        pref[col] = to01(pref[col])
    pref = pref.rename(columns={
        "email_opt_in_clean": "pref_email_opt", "sms_opt_in_clean": "pref_sms_opt",
        "preferred_channel": "pref_channel", "preferred_language": "pref_language",
        "do_not_contact_clean": "pref_dnc",
    })
    base = base.merge(
        pref[["customer_id", "pref_email_opt", "pref_sms_opt", "pref_channel",
              "pref_language", "pref_dnc", "pref_days_since_update"]],
        on="customer_id", how="left",
    )
    return base


def _window_month_flags(events: pd.DataFrame, as_of, date_col: str) -> pd.DataFrame:
    start, end = window_bounds(as_of, WINDOW_DAYS)
    months = target_months(as_of, WINDOW_DAYS)
    target_holiday = bool(months & {11, 12})
    events = events.copy()
    events["_month"] = events[date_col].dt.month
    events["_is_target_month"] = events["_month"].isin(months).astype(int)
    events["_is_target_holiday"] = (
        events["_month"].isin([11, 12]) if target_holiday else pd.Series(False, index=events.index)
    ).astype(int)
    return events, start, end, months, target_holiday


def session_features(raw: dict[str, pd.DataFrame], as_of) -> pd.DataFrame:
    s = raw["sessions"].copy()
    s["ts"] = parse_datetime(s["session_timestamp"])
    s = s[s["ts"] <= pd.Timestamp(as_of)].copy()
    s["_bounce"] = to01(s["bounce_clean"])
    s["_camp"] = (s["campaign_status"] == "Has Campaign").astype(int)

    g = s.groupby("customer_id")
    f = pd.DataFrame(index=g.size().index)
    f["sess_n"] = g.size()
    f["sess_days_since_last"] = days_since(g["ts"].max(), as_of)
    f["sess_tenure_days"] = days_since(g["ts"].min(), as_of)
    for d in (30, 90, 180, 365):
        f[f"sess_n_{d}d"] = s.assign(
            _m=s["ts"] >= pd.Timestamp(as_of) - pd.Timedelta(days=d)
        ).groupby("customer_id")["_m"].sum()
    for col, out in [("session_duration", "dur"), ("pages_viewed", "pages"),
                     ("cart_additions", "cart")]:
        f[f"sess_{out}_mean"] = g[col].mean()
        f[f"sess_{out}_max"] = g[col].max()
        f[f"sess_{out}_sum"] = g[col].sum()
    f["sess_dur_std"] = g["session_duration"].std()
    f["sess_bounce_rate"] = g["_bounce"].mean()
    f["sess_campaign_rate"] = g["_camp"].mean()
    f["sess_campaign_n"] = g["_camp"].sum()

    f = f.join(shares(s, "customer_id", "traffic_source_clean", TRAFFIC, "sess_src_"))
    f = f.join(shares(s, "customer_id", "device_type_clean", DEVICE, "sess_dev_"))

    # Legacy fixed-window features (kept so the original model keeps working).
    s["_year_month"] = s["ts"].dt.to_period("M")
    s["_is_oct"] = s["ts"].dt.month.eq(10).astype(int)
    s["_is_holiday"] = s["ts"].dt.month.isin([11, 12]).astype(int)
    s["_is_q4"] = s["ts"].dt.month.isin([10, 11, 12]).astype(int)
    seas = s.groupby("customer_id").agg(
        sess_oct_n=("_is_oct", "sum"),
        sess_holiday_n=("_is_holiday", "sum"),
        sess_q4_n=("_is_q4", "sum"),
    )
    seas["sess_oct_share"] = seas["sess_oct_n"] / f["sess_n"]
    seas["sess_holiday_share"] = seas["sess_holiday_n"] / f["sess_n"]
    seas["sess_q4_share"] = seas["sess_q4_n"] / f["sess_n"]
    monthly = s.groupby(["customer_id", "_year_month"]).size().rename("n").reset_index()
    mg = monthly.groupby("customer_id")["n"]
    seas["sess_monthly_mean"] = mg.mean()
    seas["sess_monthly_max"] = mg.max()
    seas["sess_seasonal_index"] = (
        seas["sess_monthly_max"] / seas["sess_monthly_mean"].replace(0, np.nan)
    )

    # Target-window seasonality block.
    s2, _, _, months, target_holiday = _window_month_flags(s, as_of, "ts")
    target = s2.groupby("customer_id").agg(
        sess_target_month_n=("_is_target_month", "sum"),
        sess_target_holiday_n=("_is_target_holiday", "sum"),
    )
    target["sess_target_month_share"] = target["sess_target_month_n"] / f["sess_n"]
    target["sess_target_month_active"] = (target["sess_target_month_n"] > 0).astype(int)
    target["sess_target_holiday_share"] = target["sess_target_holiday_n"] / f["sess_n"]
    target["sess_target_holiday_active"] = (target["sess_target_holiday_n"] > 0).astype(int)
    if not target_holiday:
        target["sess_target_holiday_share"] = 0.0
        target["sess_target_holiday_active"] = 0
    seas = seas.join(target)
    f = f.join(seas)

    camp = raw["campaigns"][["campaign_id", "campaign_type"]]
    sm = s[["customer_id", "campaign_id"]].dropna(subset=["campaign_id"]).copy()
    sm["campaign_id"] = sm["campaign_id"].astype("int64")
    sm = sm.merge(camp, on="campaign_id", how="left")
    f = f.join(shares(sm, "customer_id", "campaign_type", CAMPAIGN_TYPES, "sess_camptype_"))
    return f


def transaction_features(raw: dict[str, pd.DataFrame], as_of) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = raw["transactions"].copy()
    t["d"] = parse_datetime(t["transaction_timestamp"])
    t = t[t["d"] <= pd.Timestamp(as_of)].sort_values(["customer_id", "d"]).copy()
    t["_disc"] = to01(t["discount_applied_clean"])
    t["_high"] = to01(t["high_value_flag_clean"])

    g = t.groupby("customer_id")
    f = pd.DataFrame(index=g.size().index)
    f["txn_n"] = g.size()
    f["txn_days_since_last"] = days_since(g["d"].max(), as_of)
    f["txn_tenure_days"] = days_since(g["d"].min(), as_of)
    for d in (30, 90, 180, 365):
        f[f"txn_n_{d}d"] = t.assign(
            _m=t["d"] >= pd.Timestamp(as_of) - pd.Timedelta(days=d)
        ).groupby("customer_id")["_m"].sum()
        f[f"txn_spend_{d}d"] = t.assign(
            _v=np.where(t["d"] >= pd.Timestamp(as_of) - pd.Timedelta(days=d),
                        t["order_value"], 0.0)
        ).groupby("customer_id")["_v"].sum()
    for col, out in [("order_value", "value"), ("items_count", "items")]:
        f[f"txn_{out}_mean"] = g[col].mean()
        f[f"txn_{out}_max"] = g[col].max()
        f[f"txn_{out}_sum"] = g[col].sum()
    f["txn_value_std"] = g["order_value"].std()
    f["txn_value_median"] = g["order_value"].median()
    f["txn_discount_rate"] = g["_disc"].mean()
    f["txn_high_value_rate"] = g["_high"].mean()
    f["txn_last_value"] = g["order_value"].last()
    f["txn_last_items"] = g["items_count"].last()
    f["txn_spend_per_day"] = f["txn_value_sum"] / f["txn_tenure_days"].replace(0, np.nan)

    f = f.join(shares(t, "customer_id", "payment_method", PAYMENTS, "txn_pay_"))
    f = f.join(shares(t, "customer_id", "shipping_speed", SHIP_SPEEDS, "txn_ship_"))

    # Legacy fixed-window features.
    t["_year_month"] = t["d"].dt.to_period("M")
    t["_is_oct"] = t["d"].dt.month.eq(10).astype(int)
    t["_is_holiday"] = t["d"].dt.month.isin([11, 12]).astype(int)
    t["_is_q4"] = t["d"].dt.month.isin([10, 11, 12]).astype(int)
    seas = t.groupby("customer_id").agg(
        txn_oct_n=("_is_oct", "sum"),
        txn_holiday_n=("_is_holiday", "sum"),
        txn_q4_n=("_is_q4", "sum"),
    )
    seas["txn_oct_share"] = seas["txn_oct_n"] / f["txn_n"]
    seas["txn_holiday_share"] = seas["txn_holiday_n"] / f["txn_n"]
    seas["txn_q4_share"] = seas["txn_q4_n"] / f["txn_n"]
    seas["txn_oct_active"] = (seas["txn_oct_n"] > 0).astype(int)
    seas["txn_holiday_active"] = (seas["txn_holiday_n"] > 0).astype(int)
    monthly = t.groupby(["customer_id", "_year_month"]).size().rename("n").reset_index()
    mg = monthly.groupby("customer_id")["n"]
    seas["txn_monthly_mean"] = mg.mean()
    seas["txn_monthly_max"] = mg.max()
    seas["txn_monthly_std"] = mg.std()
    seas["txn_seasonal_index"] = (
        seas["txn_monthly_max"] / seas["txn_monthly_mean"].replace(0, np.nan)
    )
    seas["txn_month_cv"] = (
        seas["txn_monthly_std"] / seas["txn_monthly_mean"].replace(0, np.nan)
    )
    last_oct = t[t["_is_oct"] == 1].groupby("customer_id")["d"].max()
    seas["txn_days_since_last_oct"] = days_since(last_oct, as_of)
    f = f.join(seas)
    f["txn_dormant_holiday_affinity"] = (
        (f["txn_days_since_last"] > 180).astype(int) * f["txn_holiday_share"]
    )

    # Target-window seasonality block.
    t2, _, _, _, target_holiday = _window_month_flags(t, as_of, "d")
    target = t2.groupby("customer_id").agg(
        txn_target_month_n=("_is_target_month", "sum"),
        txn_target_holiday_n=("_is_target_holiday", "sum"),
    )
    target["txn_target_month_share"] = target["txn_target_month_n"] / f["txn_n"]
    target["txn_target_month_active"] = (target["txn_target_month_n"] > 0).astype(int)
    target["txn_target_holiday_share"] = target["txn_target_holiday_n"] / f["txn_n"]
    target["txn_target_holiday_active"] = (target["txn_target_holiday_n"] > 0).astype(int)
    if not target_holiday:
        target["txn_target_holiday_share"] = 0.0
        target["txn_target_holiday_active"] = 0
    last_target = t2[t2["_is_target_month"] == 1].groupby("customer_id")["d"].max()
    target["txn_days_since_last_target_month"] = days_since(last_target, as_of)
    f = f.join(target)
    f["txn_dormant_target_affinity"] = (
        (f["txn_days_since_last"] > 180).astype(int) * f["txn_target_month_share"]
    )
    f["txn_target_holiday_affinity"] = (
        (f["txn_days_since_last"] > 180).astype(int) * f["txn_target_holiday_share"]
    )
    return f, t


def item_features(raw: dict[str, pd.DataFrame], txn: pd.DataFrame) -> pd.DataFrame:
    items = raw["transaction_items"].copy()
    products = raw["products"][["product_id", "category", "margin_pct", "is_promotable_clean"]].copy()
    items = items.merge(products, on="product_id", how="left")
    items["line_value"] = items["quantity"] * items["unit_price"]
    items["_promo"] = to01(items["is_promotable_clean"]).fillna(0)
    items = items.merge(txn[["transaction_id", "customer_id", "d"]], on="transaction_id", how="left")
    items = items.dropna(subset=["customer_id"])
    items["customer_id"] = items["customer_id"].astype("int64")

    g = items.groupby("customer_id")
    f = pd.DataFrame(index=g.size().index)
    f["item_lines"] = g.size()
    f["item_qty"] = g["quantity"].sum()
    f["item_list_value"] = g["line_value"].sum()
    f["item_weighted_margin"] = (
        items.assign(_w=items["line_value"] * items["margin_pct"]).groupby("customer_id")["_w"].sum()
        / g["line_value"].sum().replace(0, np.nan)
    )
    low_margin = items.assign(
        _lm=np.where(items["margin_pct"] < 0.25, items["line_value"], 0.0)
    ).groupby("customer_id")["_lm"].sum()
    f["item_low_margin_share"] = low_margin / g["line_value"].sum().replace(0, np.nan)
    promo_value = items.assign(
        _pv=np.where(items["_promo"] == 1, items["line_value"], 0.0)
    ).groupby("customer_id")["_pv"].sum()
    f["item_promotable_share"] = promo_value / g["line_value"].sum().replace(0, np.nan)

    cat = items.groupby(["customer_id", "category"])["line_value"].sum().unstack(fill_value=0.0)
    cat = cat.div(cat.sum(axis=1).replace(0, np.nan), axis=0)
    cat.columns = [f"item_cat_{c}" for c in cat.columns]
    return f.join(cat)


def delivery_features(raw: dict[str, pd.DataFrame], txn: pd.DataFrame, as_of) -> pd.DataFrame:
    d = raw["deliveries"].copy()
    d["ship_date"] = parse_datetime(d["ship_date"])
    d = d[d["ship_date"] <= pd.Timestamp(as_of)].copy()
    d = d.merge(txn[["transaction_id", "customer_id", "d"]].rename(columns={"d": "txn_d"}),
                on="transaction_id", how="left")
    d = d.dropna(subset=["customer_id"])
    d["customer_id"] = d["customer_id"].astype("int64")
    d["_late"] = (d["delivery_status"] == "Delivered Late").astype(int)
    d["_lost"] = (d["delivery_status"] == "Lost").astype(int)
    d["_returned"] = (d["delivery_status"] == "Returned").astype(int)
    d["_transit"] = (d["delivery_status"] == "In Transit").astype(int)

    g = d.groupby("customer_id")
    f = pd.DataFrame(index=g.size().index)
    f["del_n"] = g.size()
    f["del_carriers"] = g["carrier"].nunique()
    f["del_late_rate"] = g["_late"].mean()
    f["del_lost_rate"] = g["_lost"].mean()
    f["del_returned_rate"] = g["_returned"].mean()
    f["del_transit_rate"] = g["_transit"].mean()
    f["del_delay_mean"] = g["delay_days"].mean()
    f["del_delay_max"] = g["delay_days"].max()
    f["del_delay_sum"] = g["delay_days"].sum()
    f["del_last_status_late"] = (
        d.sort_values(["customer_id", "txn_d"]).groupby("customer_id")["_late"].last()
    )
    return f


def support_features(raw: dict[str, pd.DataFrame], as_of) -> pd.DataFrame:
    s = raw["support_tickets"].copy()
    s["d"] = parse_datetime(s["created_date"])
    s = s[s["d"] <= pd.Timestamp(as_of)].copy()
    s["_open"] = (s["status"] == "Open").astype(int)
    s["_high"] = (s["priority"] == "High").astype(int)
    s["_low_csat"] = (s["csat_score"] <= 2).astype(int)

    g = s.groupby("customer_id")
    f = pd.DataFrame(index=g.size().index)
    f["sup_n"] = g.size()
    f["sup_open_n"] = g["_open"].sum()
    f["sup_high_n"] = g["_high"].sum()
    f["sup_low_csat_n"] = g["_low_csat"].sum()
    f["sup_days_since_last"] = days_since(g["d"].max(), as_of)
    f["sup_csat_mean"] = g["csat_score"].mean()
    f["sup_fr_mean"] = g["first_response_hours"].mean()
    f["sup_fr_max"] = g["first_response_hours"].max()
    f["sup_res_mean"] = g["resolution_hours"].mean()
    f["sup_res_max"] = g["resolution_hours"].max()
    f["sup_n_90d"] = s.assign(
        _m=s["d"] >= pd.Timestamp(as_of) - pd.Timedelta(days=90)
    ).groupby("customer_id")["_m"].sum()

    f = f.join(counts(s, "customer_id", "category", TICKET_CATS, "sup_cat_"))
    f = f.join(shares(s, "customer_id", "channel_clean", ["Live chat", "Email", "WhatsApp", "Phone"], "sup_ch_"))

    complaint_cats = ["Late delivery", "Damaged or wrong item", "Lost parcel", "Refund status"]
    recent = s[(s["d"] >= pd.Timestamp(as_of) - pd.Timedelta(days=90)) &
               (s["category"].isin(complaint_cats))]
    f["sup_recent_complaint"] = f.index.isin(recent["customer_id"]).astype(int)
    return f


def retention_features(raw: dict[str, pd.DataFrame], as_of) -> pd.DataFrame:
    r = raw["retention_contacts"].copy()
    r["d"] = parse_datetime(r["contact_date"])
    r = r[r["d"] <= pd.Timestamp(as_of)].copy()
    r["_redeemed"] = to01(r["redeemed_clean"])
    r["_p14"] = to01(r["purchased_within_14d_clean"])
    r["_no_offer"] = (r["offer_code"] == "OFF-NONE").astype(int)
    r["_v20"] = (r["offer_code"] == "OFF-V20").astype(int)

    g = r.groupby("customer_id")
    f = pd.DataFrame(index=g.size().index)
    f["ret_n"] = g.size()
    f["ret_redeemed_n"] = g["_redeemed"].sum()
    f["ret_redeem_rate"] = g["_redeemed"].mean()
    f["ret_purch14_rate"] = g["_p14"].mean()
    f["ret_no_offer_n"] = g["_no_offer"].sum()
    f["ret_v20_n"] = g["_v20"].sum()
    f["ret_days_since_last"] = days_since(g["d"].max(), as_of)
    f = f.join(shares(r, "customer_id", "channel", RET_CHANNELS, "ret_ch_"))
    return f


def region_marketing_features(raw: dict[str, pd.DataFrame], as_of) -> pd.DataFrame:
    camp = raw["campaigns"].copy()
    camp["start_date"] = parse_datetime(camp["start_date"])
    camp = camp[camp["start_date"] <= pd.Timestamp(as_of)]
    perf = raw["campaign_performance"].copy()
    perf["date"] = parse_datetime(perf["date"])
    perf = perf[perf["date"] <= pd.Timestamp(as_of)]
    agg = perf.groupby("campaign_id").agg(
        spend=("spend", "sum"), revenue=("attributed_revenue", "sum"),
        orders=("attributed_orders", "sum"), clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
    ).reset_index()
    camp = camp.merge(agg, on="campaign_id", how="left")
    reg = camp.groupby("region_target").agg(
        reg_spend=("spend", "sum"), reg_revenue=("revenue", "sum"),
        reg_orders=("orders", "sum"), reg_clicks=("clicks", "sum"),
        reg_impressions=("impressions", "sum"), reg_campaigns=("campaign_id", "count"),
        reg_budget_mean=("campaign_budget", "mean"),
    )
    reg["reg_roas"] = reg["reg_revenue"] / reg["reg_spend"].replace(0, np.nan)
    reg["reg_ctr"] = reg["reg_clicks"] / reg["reg_impressions"].replace(0, np.nan)

    recent = perf[perf["date"] >= pd.Timestamp(as_of) - pd.Timedelta(days=90)]
    recent = recent.merge(camp[["campaign_id", "region_target"]], on="campaign_id", how="left")
    reg_recent = recent.groupby("region_target")["spend"].sum().rename("reg_spend_90d")
    reg = reg.join(reg_recent)
    reg.index.name = "cust_region"
    return reg


def country_budget(raw: dict[str, pd.DataFrame]) -> pd.Series:
    rb = raw["budget"]
    return rb.groupby("country")["retention_budget_usd"].mean().rename("country_avg_weekly_budget")


# --------------------------------------------------------------------------- #
# Target-window seasonality context
# --------------------------------------------------------------------------- #
def seasonality_context(raw: dict[str, pd.DataFrame], as_of) -> dict:
    start, end = window_bounds(as_of, WINDOW_DAYS)
    months = target_months(as_of, WINDOW_DAYS)
    bf = next_black_friday(start)
    xmas = next_christmas(start)
    start_month = start.month

    tx = raw["transactions"][["transaction_id", "customer_id", "transaction_timestamp", "order_value"]].copy()
    tx["d"] = parse_datetime(tx["transaction_timestamp"])
    tx = tx[tx["d"] <= pd.Timestamp(as_of)]
    customers = raw["customers"][["customer_id", "country_code_clean", "region"]].rename(
        columns={"country_code_clean": "country", "region": "region"}
    )
    tx = tx.merge(customers, on="customer_id", how="left")
    tx["ym"] = tx["d"].dt.to_period("M").dt.to_timestamp()
    tx["year"] = tx["ym"].dt.year
    tx["month"] = tx["ym"].dt.month

    def index_by(key: str) -> pd.Series:
        monthly = tx.groupby([key, "ym", "year", "month"])["order_value"].sum().reset_index()
        yearly = monthly.groupby([key, "year"])["order_value"].mean().rename("year_mean").reset_index()
        monthly = monthly.merge(yearly, on=[key, "year"], how="left")
        monthly["revenue_index"] = monthly["order_value"] / monthly["year_mean"].replace(0, np.nan)
        hist = monthly[monthly["month"].isin(months)]
        prior = hist[hist["year"] < start.year]
        if len(prior):
            hist = prior
        idx = hist.groupby(key)["revenue_index"].median()
        return idx.fillna(1.0)

    country_index = index_by("country")
    region_index = index_by("region")
    country_index.name = "country_target_month_index"
    region_index.name = "region_target_month_index"

    scalars = {
        "target_window_start": start,
        "target_window_end": end,
        "target_month_start": int(start_month),
        "target_month_end": int(end.month),
        "target_month_sin": float(np.sin(2 * np.pi * start_month / 12)),
        "target_month_cos": float(np.cos(2 * np.pi * start_month / 12)),
        "target_is_holiday": int(bool(months & {11, 12})),
        "target_is_q4": int(bool(months & {10, 11, 12})),
        "target_days_to_black_friday": int((bf - start).days),
        "target_days_to_christmas": int((xmas - start).days),
        "target_is_black_friday_window": int(start <= bf <= end),
        "target_is_christmas_window": int(start <= xmas <= end),
        "target_peak_proximity": int(min(max((bf - start).days, 0), max((xmas - start).days, 0))),
    }
    return {"scalars": scalars, "country": country_index, "region": region_index}


# --------------------------------------------------------------------------- #
# Snapshot assembly
# --------------------------------------------------------------------------- #
def build_feature_frame(
    raw: dict[str, pd.DataFrame] | None = None,
    as_of=None,
    data_dir: str = DATA,
) -> pd.DataFrame:
    if raw is None:
        raw = load_raw(data_dir)
    as_of = resolve_as_of(as_of, data_dir) if as_of is None else pd.Timestamp(as_of).normalize()

    base = customer_base(raw, as_of)
    sess = session_features(raw, as_of)
    txn, txn_raw = transaction_features(raw, as_of)
    items = item_features(raw, txn_raw)
    deliv = delivery_features(raw, txn_raw, as_of)
    sup = support_features(raw, as_of)
    ret = retention_features(raw, as_of)
    reg = region_marketing_features(raw, as_of)
    budget = country_budget(raw)
    season = seasonality_context(raw, as_of)

    feats = base.copy()
    for block in (sess, txn, items, deliv, sup, ret):
        feats = feats.merge(block, left_on="customer_id", right_index=True, how="left")
    feats = feats.merge(reg, left_on="cust_region", right_index=True, how="left")
    feats = feats.merge(budget, left_on="cust_country", right_index=True, how="left")
    feats = feats.merge(season["country"], left_on="cust_country", right_index=True, how="left")
    feats = feats.merge(season["region"], left_on="cust_region", right_index=True, how="left")
    feats = feats.assign(**season["scalars"])

    country_idx = feats["country_target_month_index"].fillna(1.0)
    feats = feats.assign(
        country_target_month_index=country_idx,
        region_target_month_index=feats["region_target_month_index"].fillna(1.0),
        sess_target_month_weighted_share=(
            feats["sess_target_month_share"].fillna(0.0) * country_idx
        ),
        txn_target_month_weighted_share=(
            feats["txn_target_month_share"].fillna(0.0) * country_idx
        ),
    )

    count_prefixes = ("sess_n", "txn_n", "item_lines", "item_qty", "del_n", "sup_n", "ret_n",
                      "ret_redeemed_n", "ret_no_offer_n", "ret_v20_n")
    for col in feats.columns:
        if col.startswith(count_prefixes):
            feats[col] = feats[col].fillna(0)

    # Values used by the offer planner.
    expected_order = np.where(
        feats["txn_n_90d"].fillna(0) > 0,
        feats["txn_spend_90d"] / feats["txn_n_90d"].replace(0, np.nan),
        feats["txn_value_mean"],
    )
    feats = feats.assign(
        expected_order_value=pd.Series(expected_order, index=feats.index).fillna(60.0),
        margin=feats["item_weighted_margin"].fillna(0.35),
        item_promotable_share=feats["item_promotable_share"].fillna(0.0),
        _as_of=pd.Timestamp(as_of),
    )

    r = raw["retention_contacts"][["customer_id", "contact_date"]].copy()
    r["contact_date"] = parse_datetime(r["contact_date"])
    r = r[r["contact_date"] <= pd.Timestamp(as_of)]
    last = r.groupby("customer_id")["contact_date"].max().rename("last_contact_date")
    return feats.merge(last, left_on="customer_id", right_index=True, how="left")


def build_labels(
    raw: dict[str, pd.DataFrame] | None = None,
    as_of=None,
    data_dir: str = DATA,
    customer_ids=None,
) -> pd.DataFrame:
    if raw is None:
        raw = load_raw(data_dir)
    as_of = pd.Timestamp(as_of).normalize()
    start, end = window_bounds(as_of, WINDOW_DAYS)

    tx = raw["transactions"][["customer_id", "transaction_timestamp"]].copy()
    tx["d"] = parse_datetime(tx["transaction_timestamp"])
    buyers = set(tx[(tx["d"] > as_of) & (tx["d"] <= end)]["customer_id"].unique())

    if customer_ids is None:
        customer_ids = raw["customers"]["customer_id"]
    ids = pd.Series(customer_ids).drop_duplicates()
    return pd.DataFrame({
        "customer_id": ids.to_numpy(),
        "target": ids.isin(buyers).astype(int).to_numpy(),
    })


def build_snapshot(
    raw: dict[str, pd.DataFrame] | None = None,
    as_of=None,
    data_dir: str = DATA,
    labelled: bool = True,
) -> pd.DataFrame:
    if raw is None:
        raw = load_raw(data_dir)
    as_of = resolve_as_of(as_of, data_dir) if as_of is None else pd.Timestamp(as_of).normalize()
    feats = build_feature_frame(raw, as_of, data_dir)
    feats["as_of"] = pd.Timestamp(as_of)
    if not labelled:
        return feats
    labels = build_labels(raw, as_of, data_dir, feats["customer_id"])
    out = feats.merge(labels, on="customer_id", how="left")
    out["target"] = out["target"].fillna(0).astype("int8")
    return out


def snapshot_dates(
    raw: dict[str, pd.DataFrame],
    as_of=None,
    count: int = 12,
    freq: str = "monthly",
    labelled: bool = True,
) -> list[pd.Timestamp]:
    """Return snapshot as-of dates ending before the latest observable label."""
    latest = resolve_as_of(as_of, DATA) if as_of is None else pd.Timestamp(as_of).normalize()
    if labelled:
        latest = (latest - pd.Timedelta(days=WINDOW_DAYS)).normalize()
    dates = month_end_dates(latest, count) if freq == "monthly" else weekly_dates(latest, count)
    return [d for d in dates if d <= latest]


def write_snapshot(df: pd.DataFrame, out_dir: str, as_of) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"snapshot_{pd.Timestamp(as_of).date().isoformat()}.pkl")
    df.to_pickle(path)
    return path
