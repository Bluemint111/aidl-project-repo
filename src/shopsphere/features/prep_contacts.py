from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from shopsphere.common import (
    AOV_FALLBACK, BASE_FEATURES, CAMPAIGN_TYPES, CATEGORIES, CHANNELS,
    CUSTOMER_FEATURES, DATA_DIR, DELIVERY_STATUS, DEVICE, MARGIN_FALLBACK,
    MODEL_FEATURES, OUT_DATA, PAYMENTS, ROLLING_MODELS, SHIP_SPEEDS,
    SUPPORT_CHANNELS, TICKET_CATS, TIERS, TRAFFIC, counts, ensure_dirs,
    offer_catalogue, offer_cost_series, read_budget, read_csv, resolve_as_of,
    sample_ref, shares, to01, window_bounds,
)
from shopsphere.data import asof_features as af
from shopsphere.data.asof_time import parse_datetime
from shopsphere.models.scoring import RollingScorer


def days_between(as_of: pd.Series, dates: pd.Series) -> pd.Series:
    return (as_of - parse_datetime(dates)).dt.days


def base_contact_frame(contacts: pd.DataFrame) -> pd.DataFrame:
    c = read_csv("customers.csv")
    c["signup_date"] = parse_datetime(c["signup_date"])
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
    base = c[[
        "customer_id", "cust_age", "cust_country", "cust_region", "signup_date",
        "cust_loyalty", "cust_email_open", "cust_email_open_missing",
        "cust_discount_rate", "cust_review", "has_referral", "cust_tier", "tier_ord",
        "cust_credit_limit",
    ]].copy()

    geo = read_csv("geo_data.csv").rename(columns={
        "geo_ip_region": "cust_region", "average_income": "geo_income",
        "urban_ratio": "geo_urban", "internet_penetration": "geo_internet",
        "region_tier": "geo_tier",
    })
    base = base.merge(geo, on="cust_region", how="left")

    pref = read_csv("customer_preferences.csv")
    pref["last_updated"] = parse_datetime(pref["last_updated"])
    for col in ["email_opt_in_clean", "sms_opt_in_clean", "do_not_contact_clean"]:
        pref[col] = to01(pref[col])
    pref = pref.rename(columns={
        "email_opt_in_clean": "pref_email_opt", "sms_opt_in_clean": "pref_sms_opt",
        "preferred_channel": "pref_channel", "preferred_language": "pref_language",
        "do_not_contact_clean": "pref_dnc",
    })
    base = base.merge(
        pref[["customer_id", "pref_email_opt", "pref_sms_opt", "pref_channel",
              "pref_language", "pref_dnc", "last_updated"]],
        on="customer_id", how="left",
    )

    out = contacts.merge(base, on="customer_id", how="left")
    out["tenure_days"] = days_between(out["contact_date"], out["signup_date"])
    out["pref_days_since_update"] = days_between(out["contact_date"], out["last_updated"])
    out = out.drop(columns=["signup_date", "last_updated"])

    budget = read_budget().groupby("country")["retention_budget_usd"].mean()
    out["country_avg_weekly_budget"] = out["cust_country"].map(budget)

    offers = offer_catalogue()
    offers = offers.rename(columns={
        "value": "offer_value", "requires_promotable": "offer_requires_promotable",
        "counts_monthly": "offer_counts_monthly",
    })
    out = out.merge(
        offers[["offer_code", "offer_type", "offer_value", "offer_requires_promotable",
                "offer_counts_monthly", "eligible_tiers", "tier_list"]],
        on="offer_code", how="left",
    )

    def tier_match(tier, tier_list) -> int:
        if isinstance(tier_list, str):
            tier_list = [tier_list]
        if tier_list is None:
            return 0
        names = [str(t).strip() for t in tier_list]
        if any(t.lower() == "all" for t in names):
            return 1
        return int(str(tier) in names)

    out["tier_offer_match"] = [
        tier_match(t, v) for t, v in zip(out["cust_tier"], out["tier_list"])
    ]
    out["contact_channel_is_preferred"] = (
        out["channel"].astype(str) == out["pref_channel"].astype(str)
    ).astype(int)
    out = out.drop(columns=["eligible_tiers", "tier_list"])
    return out


def session_block(contacts: pd.DataFrame) -> pd.DataFrame:
    s = read_csv(
        "shopsphere_sessions_cleaned.csv",
        usecols=["customer_id", "session_timestamp", "session_duration", "pages_viewed",
                 "cart_additions", "campaign_id", "traffic_source_clean",
                 "device_type_clean", "bounce_clean", "campaign_status"],
    )
    s["event_date"] = parse_datetime(s["session_timestamp"]).dt.normalize()
    s["_bounce"] = to01(s["bounce_clean"])
    s["_camp"] = (s["campaign_status"] == "Has Campaign").astype(int)
    m = contacts[["contact_id", "customer_id", "contact_date"]].merge(s, on="customer_id", how="inner")
    m = m[m["event_date"] <= m["contact_date"]].copy()
    m["days_before"] = (m["contact_date"] - m["event_date"]).dt.days

    g = m.groupby("contact_id")
    f = pd.DataFrame(index=g.size().index)
    f["sess_n"] = g.size()
    f["sess_days_since_last"] = g["days_before"].min()
    f["sess_tenure_days"] = g["days_before"].max()
    for d in (30, 90, 180, 365):
        f[f"sess_n_{d}d"] = m.assign(_m=m["days_before"] <= d).groupby("contact_id")["_m"].sum()
    for col, out in [("session_duration", "dur"), ("pages_viewed", "pages"),
                     ("cart_additions", "cart")]:
        f[f"sess_{out}_mean"] = g[col].mean()
        f[f"sess_{out}_max"] = g[col].max()
        f[f"sess_{out}_sum"] = g[col].sum()
    f["sess_dur_std"] = g["session_duration"].std()
    f["sess_bounce_rate"] = g["_bounce"].mean()
    f["sess_campaign_rate"] = g["_camp"].mean()
    f["sess_campaign_n"] = g["_camp"].sum()
    f = f.join(shares(m, "contact_id", "traffic_source_clean", TRAFFIC, "sess_src_"))
    f = f.join(shares(m, "contact_id", "device_type_clean", DEVICE, "sess_dev_"))

    camp = read_csv("marketing_campaigns.csv", usecols=["campaign_id", "campaign_type"])
    sm = m[["contact_id", "campaign_id"]].dropna(subset=["campaign_id"]).copy()
    sm["campaign_id"] = sm["campaign_id"].astype("int64")
    sm = sm.merge(camp, on="campaign_id", how="left")
    f = f.join(shares(sm, "contact_id", "campaign_type", CAMPAIGN_TYPES, "sess_camptype_"))
    return f


def transaction_block(contacts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = read_csv(
        "transactions.csv",
        usecols=["transaction_id", "customer_id", "transaction_timestamp", "order_value",
                 "items_count", "payment_method", "discount_applied_clean",
                 "shipping_speed", "high_value_flag_clean"],
    )
    t["event_date"] = parse_datetime(t["transaction_timestamp"]).dt.normalize()
    t["_disc"] = to01(t["discount_applied_clean"])
    t["_high"] = to01(t["high_value_flag_clean"])
    m = contacts[["contact_id", "customer_id", "contact_date"]].merge(t, on="customer_id", how="inner")
    m = m[m["event_date"] <= m["contact_date"]].copy()
    m["days_before"] = (m["contact_date"] - m["event_date"]).dt.days

    g = m.groupby("contact_id")
    f = pd.DataFrame(index=g.size().index)
    f["txn_n"] = g.size()
    f["txn_days_since_last"] = g["days_before"].min()
    f["txn_tenure_days"] = g["days_before"].max()
    for d in (30, 90, 180, 365):
        f[f"txn_n_{d}d"] = m.assign(_m=m["days_before"] <= d).groupby("contact_id")["_m"].sum()
        f[f"txn_spend_{d}d"] = m.assign(
            _v=np.where(m["days_before"] <= d, m["order_value"], 0.0)
        ).groupby("contact_id")["_v"].sum()
    for col, out in [("order_value", "value"), ("items_count", "items")]:
        f[f"txn_{out}_mean"] = g[col].mean()
        f[f"txn_{out}_max"] = g[col].max()
        f[f"txn_{out}_sum"] = g[col].sum()
    f["txn_value_std"] = g["order_value"].std()
    f["txn_value_median"] = g["order_value"].median()
    f["txn_discount_rate"] = g["_disc"].mean()
    f["txn_high_value_rate"] = g["_high"].mean()
    last = m.sort_values(["contact_id", "event_date"]).groupby("contact_id")
    f["txn_last_value"] = last["order_value"].last()
    f["txn_last_items"] = last["items_count"].last()
    f["txn_spend_per_day"] = f["txn_value_sum"] / f["txn_tenure_days"].replace(0, np.nan)
    f = f.join(shares(m, "contact_id", "payment_method", PAYMENTS, "txn_pay_"))
    f = f.join(shares(m, "contact_id", "shipping_speed", SHIP_SPEEDS, "txn_ship_"))
    return f, m


def item_block(txn_events: pd.DataFrame) -> pd.DataFrame:
    items = read_csv("transaction_items.csv")
    products = read_csv(
        "products.csv", usecols=["product_id", "category", "margin_pct", "is_promotable_clean"]
    )
    items = items.merge(products, on="product_id", how="left")
    items["line_value"] = items["quantity"] * items["unit_price"]
    items["_promo"] = to01(items["is_promotable_clean"]).fillna(0)
    m = txn_events[["contact_id", "transaction_id", "event_date", "days_before"]].merge(
        items, on="transaction_id", how="inner"
    )

    g = m.groupby("contact_id")
    f = pd.DataFrame(index=g.size().index)
    f["item_lines"] = g.size()
    f["item_qty"] = g["quantity"].sum()
    f["item_list_value"] = g["line_value"].sum()
    f["item_weighted_margin"] = (
        m.assign(_w=m["line_value"] * m["margin_pct"]).groupby("contact_id")["_w"].sum()
        / g["line_value"].sum().replace(0, np.nan)
    )
    low_margin = m.assign(
        _lm=np.where(m["margin_pct"] < 0.25, m["line_value"], 0.0)
    ).groupby("contact_id")["_lm"].sum()
    f["item_low_margin_share"] = low_margin / g["line_value"].sum().replace(0, np.nan)
    promo_value = m.assign(
        _pv=np.where(m["_promo"] == 1, m["line_value"], 0.0)
    ).groupby("contact_id")["_pv"].sum()
    f["item_promotable_share"] = promo_value / g["line_value"].sum().replace(0, np.nan)

    cat = m.groupby(["contact_id", "category"])["line_value"].sum().unstack(fill_value=0.0)
    cat = cat.reindex(columns=CATEGORIES, fill_value=0.0)
    cat = cat.div(cat.sum(axis=1).replace(0, np.nan), axis=0)
    cat.columns = [f"item_cat_{c}" for c in cat.columns]
    return f.join(cat)


def delivery_block(txn_events: pd.DataFrame) -> pd.DataFrame:
    d = read_csv(
        "deliveries.csv",
        usecols=["transaction_id", "carrier", "ship_date", "delivered_date", "delay_days",
                 "delivery_status"],
    )
    d["ship_date"] = parse_datetime(d["ship_date"])
    m = txn_events[["contact_id", "transaction_id", "contact_date", "event_date"]].merge(
        d, on="transaction_id", how="inner"
    )
    m = m[m["ship_date"] <= m["contact_date"]].copy()
    m["_late"] = (m["delivery_status"] == "Delivered Late").astype(int)
    m["_lost"] = (m["delivery_status"] == "Lost").astype(int)
    m["_returned"] = (m["delivery_status"] == "Returned").astype(int)
    m["_transit"] = (m["delivery_status"] == "In Transit").astype(int)

    g = m.groupby("contact_id")
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
        m.sort_values(["contact_id", "ship_date"]).groupby("contact_id")["_late"].last()
    )
    return f


def support_block(contacts: pd.DataFrame) -> pd.DataFrame:
    s = read_csv(
        "support_tickets.csv",
        usecols=["ticket_id", "customer_id", "created_date", "channel_clean", "category",
                 "priority", "status", "first_response_hours", "resolution_hours", "csat_score"],
    )
    s["event_date"] = parse_datetime(s["created_date"]).dt.normalize()
    s["_open"] = (s["status"] == "Open").astype(int)
    s["_high"] = (s["priority"] == "High").astype(int)
    s["_low_csat"] = (s["csat_score"] <= 2).astype(int)
    m = contacts[["contact_id", "customer_id", "contact_date"]].merge(s, on="customer_id", how="inner")
    m = m[m["event_date"] <= m["contact_date"]].copy()
    m["days_before"] = (m["contact_date"] - m["event_date"]).dt.days

    g = m.groupby("contact_id")
    f = pd.DataFrame(index=g.size().index)
    f["sup_n"] = g.size()
    f["sup_open_n"] = g["_open"].sum()
    f["sup_high_n"] = g["_high"].sum()
    f["sup_low_csat_n"] = g["_low_csat"].sum()
    f["sup_days_since_last"] = g["days_before"].min()
    f["sup_csat_mean"] = g["csat_score"].mean()
    f["sup_fr_mean"] = g["first_response_hours"].mean()
    f["sup_fr_max"] = g["first_response_hours"].max()
    f["sup_res_mean"] = g["resolution_hours"].mean()
    f["sup_res_max"] = g["resolution_hours"].max()
    f["sup_n_90d"] = m.assign(_m=m["days_before"] <= 90).groupby("contact_id")["_m"].sum()

    complaint_cats = ["Late delivery", "Damaged or wrong item", "Lost parcel", "Refund status"]
    recent = m[(m["days_before"] <= 90) & (m["category"].isin(complaint_cats))]
    f["sup_recent_complaint"] = f.index.isin(recent["contact_id"]).astype(int)
    f = f.join(counts(m, "contact_id", "category", TICKET_CATS, "sup_cat_"))
    f = f.join(shares(m, "contact_id", "channel_clean", SUPPORT_CHANNELS, "sup_ch_"))
    return f


def historical_features() -> pd.DataFrame:
    contacts = read_csv("retention_contact_history.csv")
    contacts["contact_date"] = parse_datetime(contacts["contact_date"])
    contacts["purchased_within_14d"] = to01(contacts["purchased_within_14d_clean"]).fillna(0).astype(int)
    contacts["redeemed"] = to01(contacts["redeemed_clean"])

    feats = base_contact_frame(contacts)
    print("  sessions")
    sess = session_block(contacts)
    print("  transactions")
    txn, txn_events = transaction_block(contacts)
    print("  items")
    items = item_block(txn_events)
    print("  deliveries")
    deliv = delivery_block(txn_events)
    print("  support")
    sup = support_block(contacts)

    feats = feats.merge(txn, left_on="contact_id", right_index=True, how="left")
    feats["expected_order_value"] = np.where(
        feats["txn_n_90d"].fillna(0) > 0,
        feats["txn_spend_90d"] / feats["txn_n_90d"].replace(0, np.nan),
        feats["txn_value_mean"],
    )
    feats["expected_order_value"] = feats["expected_order_value"].fillna(AOV_FALLBACK)
    feats["offer_cost"] = offer_cost_series(feats)

    for block in (sess, items, deliv, sup):
        feats = feats.merge(block, left_on="contact_id", right_index=True, how="left")

    feats["margin"] = feats["item_weighted_margin"].fillna(MARGIN_FALLBACK)
    feats["item_promotable_share"] = feats["item_promotable_share"].fillna(0.0)

    count_cols = [
        "sess_n", "sess_n_30d", "sess_n_90d", "sess_n_180d", "sess_n_365d",
        "sess_campaign_n", "txn_n", "txn_n_30d", "txn_n_90d", "txn_n_180d",
        "txn_n_365d", "item_lines", "item_qty", "del_n", "sup_n", "sup_open_n",
        "sup_high_n", "sup_low_csat_n", "sup_n_90d",
    ] + [f"sup_cat_{c}" for c in TICKET_CATS]
    for col in count_cols:
        if col in feats.columns:
            feats[col] = feats[col].fillna(0)
    for col in ["sup_recent_complaint"]:
        if col in feats.columns:
            feats[col] = feats[col].fillna(0).astype(int)

    for col in MODEL_FEATURES:
        if col not in feats.columns:
            feats[col] = np.nan
    feats["offer_requires_promotable"] = feats["offer_requires_promotable"].fillna(0).astype(int)
    feats["offer_counts_monthly"] = feats["offer_counts_monthly"].fillna(0).astype(int)
    feats["tier_offer_match"] = feats["tier_offer_match"].fillna(0).astype(int)
    feats["contact_channel_is_preferred"] = feats["contact_channel_is_preferred"].fillna(0).astype(int)
    return feats


def rolling_live_snapshot() -> pd.DataFrame:
    path = os.path.join(OUT_DATA, "live_scores.pkl")
    if os.path.exists(path):
        print(f"  using rolling live scores from {path}")
        return pd.read_pickle(path)

    as_of = resolve_as_of(None, DATA_DIR)
    raw = af.load_raw(DATA_DIR)
    cur = af.build_snapshot(raw, as_of, DATA_DIR, labelled=False)
    scorer = RollingScorer()
    cur["w1_probability"] = scorer.predict_proba(cur)
    cur["w1_target"] = (cur["w1_probability"] >= scorer.threshold).astype(int)
    cur["w1_as_of"] = pd.Timestamp(as_of)
    start, end = window_bounds(as_of, af.WINDOW_DAYS)
    cur["w1_window_start"] = start
    cur["w1_window_end"] = end
    return cur


def current_snapshot() -> pd.DataFrame:
    cur = rolling_live_snapshot()
    cur["sample_ref"] = sample_ref(cur["customer_id"])

    as_of = pd.Timestamp(cur["w1_as_of"].iloc[0]) if "w1_as_of" in cur else \
        resolve_as_of(None, DATA_DIR)
    r = read_csv("retention_contact_history.csv", usecols=["customer_id", "contact_date"])
    r["contact_date"] = parse_datetime(r["contact_date"])
    r = r[r["contact_date"] <= as_of]
    last = r.groupby("customer_id")["contact_date"].max().rename("last_contact_date")
    cur = cur.drop(columns=["last_contact_date"], errors="ignore")
    return cur.merge(last, left_on="customer_id", right_index=True, how="left")


def main() -> None:
    ensure_dirs()
    print("Building historical contact features ...")
    hist = historical_features()
    hist.to_pickle(os.path.join(OUT_DATA, "historical_contacts.pkl"))

    print("Building current customer snapshot ...")
    cur = current_snapshot()
    cur.to_pickle(os.path.join(OUT_DATA, "current_customers.pkl"))

    offers = offer_catalogue().drop(columns=["tier_list"])
    offers.to_csv(os.path.join(OUT_DATA, "offer_catalogue_clean.csv"), index=False)

    rolling_meta_path = os.path.join(ROLLING_MODELS, "rolling_meta.json")
    rolling_meta = {}
    if os.path.exists(rolling_meta_path):
        with open(rolling_meta_path, encoding="utf-8") as fh:
            rolling_meta = json.load(fh)
    score_as_of = cur["w1_as_of"].iloc[0] if "w1_as_of" in cur else None
    score_window_start = cur["w1_window_start"].iloc[0] if "w1_window_start" in cur else None
    score_window_end = cur["w1_window_end"].iloc[0] if "w1_window_end" in cur else None
    summary = {
        "historical_contacts": int(len(hist)),
        "historical_customers": int(hist["customer_id"].nunique()),
        "historical_purchase_rate": round(float(hist["purchased_within_14d"].mean()), 5),
        "current_customers": int(len(cur)),
        "current_w1_mean": round(float(cur["w1_probability"].mean()), 5),
        "current_w1_high_share": round(float(cur["w1_target"].mean()), 5),
        "score_as_of": str(pd.Timestamp(score_as_of).date()) if score_as_of is not None else None,
        "score_window_start": str(pd.Timestamp(score_window_start).date()) if score_window_start is not None else None,
        "score_window_end": str(pd.Timestamp(score_window_end).date()) if score_window_end is not None else None,
        "w1_model": rolling_meta.get("selected_model"),
        "w1_threshold": rolling_meta.get("selected_threshold"),
        "model_features": len(MODEL_FEATURES),
        "categorical_features": [c for c in MODEL_FEATURES if c in
                                  ["cust_country", "cust_region", "cust_tier", "geo_tier",
                                   "pref_channel", "pref_language", "offer_code", "channel",
                                   "offer_type"]],
    }
    with open(os.path.join(OUT_DATA, "prep_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
