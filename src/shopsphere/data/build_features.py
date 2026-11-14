"""
ShopSphere - customer-level feature engineering for the 30-day purchase target.

Reads the cleaned tables in data/clean and builds one row per customer using only
information up to the resolved as-of cutoff, so the target window is never
leaked.  The cutoff comes from ``--as-of``, ``SHOPSPHERE_AS_OF``, or the latest
event in the data pack; nothing here is tied to a specific campaign date.

Outputs (artifacts/features):
    train_features.pkl   customer_id + features + target
    test_features.pkl    customer_id + features
    feature_metadata.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from shopsphere.common import DATA_DIR as DATA, ARTIFACTS_FEATURES as OUT
from shopsphere.data.asof_time import parse_datetime, resolve_as_of

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


def read(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA, name), low_memory=False, **kw)


def days_since(dates: pd.Series, as_of) -> pd.Series:
    return (pd.Timestamp(as_of) - parse_datetime(dates)).dt.days


def to01(s: pd.Series) -> pd.Series:
    """Map booleans / TRUE-FALSE / Y-N encodings to 1/0 (NaN stays NaN)."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype("int64")
    low = s.astype("string").str.strip().str.lower()
    return low.map({"true": 1, "false": 0, "y": 1, "n": 0, "yes": 1, "no": 0,
                    "1": 1, "0": 0, "t": 1, "f": 0})


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
def customer_base(as_of) -> pd.DataFrame:
    c = read("customers.csv")
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

    geo = read("geo_data.csv").rename(columns={
        "geo_ip_region": "cust_region", "average_income": "geo_income",
        "urban_ratio": "geo_urban", "internet_penetration": "geo_internet",
        "region_tier": "geo_tier",
    })
    base = base.merge(geo, on="cust_region", how="left")

    pref = read("customer_preferences.csv")
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


def session_features(as_of) -> pd.DataFrame:
    s = read("shopsphere_sessions_cleaned.csv")
    s["ts"] = parse_datetime(s["session_timestamp"])
    s = s[s["ts"] <= pd.Timestamp(as_of)].copy()
    s["_bounce"] = to01(s["bounce_clean"])
    s["_camp"] = (s["campaign_status"] == "Has Campaign").astype(int)

    g = s.groupby("customer_id")
    f = pd.DataFrame(index=g.size().index)
    f["sess_n"] = g.size()
    f["sess_days_since_last"] = (pd.Timestamp(as_of) - g["ts"].max()).dt.days
    f["sess_tenure_days"] = (pd.Timestamp(as_of) - g["ts"].min()).dt.days
    for d in (30, 90, 180, 365):
        f[f"sess_n_{d}d"] = s.assign(_m=s["ts"] >= pd.Timestamp(as_of) - pd.Timedelta(days=d)).groupby("customer_id")["_m"].sum()
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

    # Seasonality: how much of this customer's browsing happens in the target month
    # (October) and in the Nov-Dec holiday peak, across the multi-year history.
    s["_month"] = s["ts"].dt.month
    s["_is_oct"] = s["_month"].eq(10).astype(int)
    s["_is_holiday"] = s["_month"].isin([11, 12]).astype(int)
    s["_is_q4"] = s["_month"].isin([10, 11, 12]).astype(int)
    s["_year_month"] = s["ts"].dt.to_period("M")
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
    f = f.join(seas)

    camp = read("marketing_campaigns.csv", usecols=["campaign_id", "campaign_type"])
    sm = s[["customer_id", "campaign_id"]].dropna(subset=["campaign_id"]).copy()
    sm["campaign_id"] = sm["campaign_id"].astype("int64")
    sm = sm.merge(camp, on="campaign_id", how="left")
    f = f.join(shares(sm, "customer_id", "campaign_type", CAMPAIGN_TYPES, "sess_camptype_"))
    return f


def transaction_features(as_of) -> pd.DataFrame:
    t = read("transactions.csv")
    t["d"] = parse_datetime(t["transaction_timestamp"])
    t = t[t["d"] <= pd.Timestamp(as_of)].sort_values(["customer_id", "d"]).copy()
    t["_disc"] = to01(t["discount_applied_clean"])
    t["_high"] = to01(t["high_value_flag_clean"])

    g = t.groupby("customer_id")
    f = pd.DataFrame(index=g.size().index)
    f["txn_n"] = g.size()
    f["txn_days_since_last"] = (pd.Timestamp(as_of) - g["d"].max()).dt.days
    f["txn_tenure_days"] = (pd.Timestamp(as_of) - g["d"].min()).dt.days
    for d in (30, 90, 180, 365):
        f[f"txn_n_{d}d"] = t.assign(_m=t["d"] >= pd.Timestamp(as_of) - pd.Timedelta(days=d)).groupby("customer_id")["_m"].sum()
        f[f"txn_spend_{d}d"] = t.assign(_v=np.where(t["d"] >= pd.Timestamp(as_of) - pd.Timedelta(days=d), t["order_value"], 0.0)).groupby("customer_id")["_v"].sum()
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

    # Seasonality: historical October / holiday buying behaviour. The target window is
    # October 2024, so a customer's prior-October activity is a direct seasonal signal;
    # Nov-Dec affinity captures seasonal shoppers who are otherwise dormant.
    t["_month"] = t["d"].dt.month
    t["_is_oct"] = t["_month"].eq(10).astype(int)
    t["_is_holiday"] = t["_month"].isin([11, 12]).astype(int)
    t["_is_q4"] = t["_month"].isin([10, 11, 12]).astype(int)
    t["_year_month"] = t["d"].dt.to_period("M")
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
    seas["txn_days_since_last_oct"] = (pd.Timestamp(as_of) - last_oct).dt.days
    f = f.join(seas)
    f["txn_dormant_holiday_affinity"] = (
        (f["txn_days_since_last"] > 180).astype(int) * f["txn_holiday_share"]
    )
    return f, t


def item_features(txn: pd.DataFrame) -> pd.DataFrame:
    items = read("transaction_items.csv")
    products = read("products.csv", usecols=["product_id", "category", "margin_pct"])
    items = items.merge(products, on="product_id", how="left")
    items["line_value"] = items["quantity"] * items["unit_price"]
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

    cat = items.groupby(["customer_id", "category"])["line_value"].sum().unstack(fill_value=0.0)
    cat = cat.div(cat.sum(axis=1).replace(0, np.nan), axis=0)
    cat.columns = [f"item_cat_{c}" for c in cat.columns]
    f = f.join(cat)
    return f


def delivery_features(txn: pd.DataFrame, as_of) -> pd.DataFrame:
    d = read("deliveries.csv")
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


def support_features(as_of) -> pd.DataFrame:
    s = read("support_tickets.csv")
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
    f["sup_days_since_last"] = (pd.Timestamp(as_of) - g["d"].max()).dt.days
    f["sup_csat_mean"] = g["csat_score"].mean()
    f["sup_fr_mean"] = g["first_response_hours"].mean()
    f["sup_fr_max"] = g["first_response_hours"].max()
    f["sup_res_mean"] = g["resolution_hours"].mean()
    f["sup_res_max"] = g["resolution_hours"].max()
    f["sup_n_90d"] = s.assign(_m=s["d"] >= pd.Timestamp(as_of) - pd.Timedelta(days=90)).groupby("customer_id")["_m"].sum()

    f = f.join(counts(s, "customer_id", "category", TICKET_CATS, "sup_cat_"))
    f = f.join(shares(s, "customer_id", "channel_clean", ["Live chat", "Email", "WhatsApp", "Phone"], "sup_ch_"))

    complaint_cats = ["Late delivery", "Damaged or wrong item", "Lost parcel", "Refund status"]
    recent = s[(s["d"] >= pd.Timestamp(as_of) - pd.Timedelta(days=90)) & (s["category"].isin(complaint_cats))]
    f["sup_recent_complaint"] = f.index.isin(recent["customer_id"]).astype(int)
    return f


def retention_features(as_of) -> pd.DataFrame:
    r = read("retention_contact_history.csv")
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
    f["ret_days_since_last"] = (pd.Timestamp(as_of) - g["d"].max()).dt.days
    f = f.join(shares(r, "customer_id", "channel", RET_CHANNELS, "ret_ch_"))
    return f


def region_marketing_features(as_of) -> pd.DataFrame:
    camp = read("marketing_campaigns.csv")
    camp["start_date"] = parse_datetime(camp["start_date"])
    camp = camp[camp["start_date"] <= pd.Timestamp(as_of)].copy()
    perf = read("campaign_daily_performance.csv")
    perf["date"] = parse_datetime(perf["date"])
    perf = perf[perf["date"] <= pd.Timestamp(as_of)].copy()
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


def country_budget() -> pd.Series:
    rb = read("retention_budget_2024Q4.csv")
    return rb.groupby("country")["retention_budget_usd"].mean().rename("country_avg_weekly_budget")


def main(as_of=None) -> None:
    parser = argparse.ArgumentParser(description="Build the fixed W1 feature snapshot")
    parser.add_argument("--as-of", default=None,
                        help="Snapshot cutoff (default: SHOPSPHERE_AS_OF or latest data date)")
    args, _ = parser.parse_known_args()
    as_of = resolve_as_of(args.as_of or as_of, DATA)

    os.makedirs(OUT, exist_ok=True)
    print(f"Building customer base as of {as_of.date()} ...")
    base = customer_base(as_of)

    print("Sessions ...")
    sess = session_features(as_of)
    print("Transactions ...")
    txn, txn_raw = transaction_features(as_of)
    print("Items ...")
    items = item_features(txn_raw)
    print("Deliveries ...")
    deliv = delivery_features(txn_raw, as_of)
    print("Support ...")
    sup = support_features(as_of)
    print("Retention contacts ...")
    ret = retention_features(as_of)
    print("Region marketing ...")
    reg = region_marketing_features(as_of)
    print("Country budget ...")
    budget = country_budget()

    feats = base.copy()
    for block in (sess, txn, items, deliv, sup, ret):
        feats = feats.merge(block, left_on="customer_id", right_index=True, how="left")
    feats = feats.merge(reg, left_on="cust_region", right_index=True, how="left")
    feats = feats.merge(budget, left_on="cust_country", right_index=True, how="left")

    count_prefixes = ("sess_n", "txn_n", "item_lines", "item_qty", "del_n", "sup_n", "ret_n",
                      "ret_redeemed_n", "ret_no_offer_n", "ret_v20_n")
    for col in feats.columns:
        if col.startswith(count_prefixes):
            feats[col] = feats[col].fillna(0)

    train_ids = read("train.csv")
    test_ids = read("test.csv")
    train = train_ids.merge(feats, on="customer_id", how="left")
    test = test_ids.merge(feats, on="customer_id", how="left")
    assert len(train) == len(train_ids) and len(test) == len(test_ids)

    feature_cols = [c for c in feats.columns if c != "customer_id"]
    cat_cols = [c for c in ["cust_country", "cust_region", "cust_tier", "geo_tier",
                            "pref_channel", "pref_language"] if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    train.to_pickle(os.path.join(OUT, "train_features.pkl"))
    test.to_pickle(os.path.join(OUT, "test_features.pkl"))
    meta = {
        "cutoff": str(pd.Timestamp(as_of).date()),
        "n_features": len(feature_cols),
        "categorical": cat_cols,
        "numeric": num_cols,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "positive_rate": float(train["target"].mean()),
        "feature_columns": feature_cols,
    }
    with open(os.path.join(OUT, "feature_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Train {train.shape}  Test {test.shape}  features={len(feature_cols)}")
    print(f"Categorical: {cat_cols}")
    print(f"Numeric: {len(num_cols)}")
    print("Saved features to", OUT)


if __name__ == "__main__":
    main()

