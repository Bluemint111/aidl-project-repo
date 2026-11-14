from __future__ import annotations

import hashlib
import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(BASE))
SRC_DIR = os.path.join(PROJECT, "src")
DATA_DIR = os.environ.get("SHOPSPHERE_DATA_DIR", os.path.join(PROJECT, "data", "clean"))
DATA_PROCESSED = os.path.join(PROJECT, "data", "processed")
ARTIFACTS = os.path.join(PROJECT, "artifacts")
ARTIFACTS_FEATURES = os.path.join(ARTIFACTS, "features")
ARTIFACTS_SNAPSHOTS = os.path.join(ARTIFACTS, "snapshots")
ARTIFACTS_MODELS = os.path.join(ARTIFACTS, "models")
FIXED_MODELS = os.path.join(ARTIFACTS_MODELS, "fixed")
ROLLING_MODELS = os.path.join(ARTIFACTS_MODELS, "rolling")
OFFER_MODELS = os.path.join(ARTIFACTS_MODELS, "offer")
ARTIFACTS_REPORTS = os.path.join(ARTIFACTS, "reports")
FIXED_REPORTS = os.path.join(ARTIFACTS_REPORTS, "fixed")
ROLLING_REPORTS = os.path.join(ARTIFACTS_REPORTS, "rolling")
OFFER_REPORTS = os.path.join(ARTIFACTS_REPORTS, "offer")
ANALYTICS_REPORTS = os.path.join(ARTIFACTS_REPORTS, "analytics")
PLAN_REPORTS = os.path.join(ARTIFACTS_REPORTS, "plans")
LOG_REPORTS = os.path.join(ARTIFACTS_REPORTS, "logs")
OUT_DATA = DATA_PROCESSED
OUT_MODELS = OFFER_MODELS
OUT_REPORTS = ARTIFACTS_REPORTS
OUT_OUTPUTS = os.path.join(ARTIFACTS, "outputs")
DB_PATH = os.path.join(DATA_PROCESSED, "approvals.sqlite")
MARGIN_FALLBACK = 0.35
AOV_FALLBACK = 60.0

sys.path.insert(0, SRC_DIR)
from shopsphere.data.asof_time import resolve_as_of, window_bounds  # noqa: E402

CATEGORIES = [
    "Beauty", "Books & Media", "Electronics", "Fashion",
    "Groceries", "Home & Living", "Sports & Outdoors", "Toys & Kids",
]
TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
CHANNELS = ["email", "app_push", "sms"]
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
SUPPORT_CHANNELS = ["Live chat", "Email", "WhatsApp", "Phone"]

BASE_FEATURES = [
    "cust_age", "cust_country", "cust_region", "tenure_days", "cust_loyalty",
    "cust_email_open", "cust_email_open_missing", "cust_discount_rate", "cust_review",
    "has_referral", "cust_tier", "tier_ord", "cust_credit_limit", "geo_income",
    "geo_urban", "geo_internet", "geo_tier", "pref_email_opt", "pref_sms_opt",
    "pref_channel", "pref_language", "pref_dnc", "pref_days_since_update",
    "country_avg_weekly_budget",
]
SESSION_FEATURES = [
    "sess_n", "sess_days_since_last", "sess_tenure_days", "sess_n_30d", "sess_n_90d",
    "sess_n_180d", "sess_n_365d", "sess_dur_mean", "sess_dur_max", "sess_dur_sum",
    "sess_pages_mean", "sess_pages_max", "sess_pages_sum", "sess_cart_mean",
    "sess_cart_max", "sess_cart_sum", "sess_dur_std", "sess_bounce_rate",
    "sess_campaign_rate", "sess_campaign_n", "sess_src_organic", "sess_src_direct",
    "sess_src_ads", "sess_src_email", "sess_src_social", "sess_dev_mobile",
    "sess_dev_desktop", "sess_dev_tablet", "sess_camptype_Email",
    "sess_camptype_Search", "sess_camptype_Display", "sess_camptype_Social",
    "sess_camptype_Affiliate",
]
TXN_FEATURES = [
    "txn_n", "txn_days_since_last", "txn_tenure_days", "txn_n_30d", "txn_spend_30d",
    "txn_n_90d", "txn_spend_90d", "txn_n_180d", "txn_spend_180d", "txn_n_365d",
    "txn_spend_365d", "txn_value_mean", "txn_value_max", "txn_value_sum",
    "txn_items_mean", "txn_items_max", "txn_items_sum", "txn_value_std",
    "txn_value_median", "txn_discount_rate", "txn_high_value_rate", "txn_last_value",
    "txn_last_items", "txn_spend_per_day", "txn_pay_credit_card", "txn_pay_debit_card",
    "txn_pay_paypal", "txn_pay_crypto", "txn_ship_standard", "txn_ship_express",
    "txn_ship_overnight",
]
ITEM_FEATURES = [
    "item_lines", "item_qty", "item_list_value", "item_weighted_margin",
    "item_low_margin_share", "item_promotable_share",
] + [f"item_cat_{c}" for c in CATEGORIES]
DELIVERY_FEATURES = [
    "del_n", "del_carriers", "del_late_rate", "del_lost_rate", "del_returned_rate",
    "del_transit_rate", "del_delay_mean", "del_delay_max", "del_delay_sum",
    "del_last_status_late",
]
SUPPORT_FEATURES = [
    "sup_n", "sup_open_n", "sup_high_n", "sup_low_csat_n", "sup_days_since_last",
    "sup_csat_mean", "sup_fr_mean", "sup_fr_max", "sup_res_mean", "sup_res_max",
    "sup_n_90d", "sup_recent_complaint",
] + [f"sup_cat_{c}" for c in TICKET_CATS] + [f"sup_ch_{c}" for c in SUPPORT_CHANNELS]
OFFER_FEATURES = [
    "offer_code", "channel", "offer_type", "offer_value", "offer_requires_promotable",
    "offer_counts_monthly", "offer_cost", "contact_channel_is_preferred",
    "tier_offer_match",
]
VALUE_FEATURES = ["expected_order_value", "margin"]

CUSTOMER_FEATURES = BASE_FEATURES + SESSION_FEATURES + TXN_FEATURES + ITEM_FEATURES + \
    DELIVERY_FEATURES + SUPPORT_FEATURES + VALUE_FEATURES

MODEL_FEATURES = CUSTOMER_FEATURES + OFFER_FEATURES

CATEGORICAL_FEATURES = [
    "cust_country", "cust_region", "cust_tier", "geo_tier", "pref_channel",
    "pref_language", "offer_code", "channel", "offer_type",
]


def ensure_dirs() -> None:
    for path in (
        DATA_PROCESSED, ARTIFACTS_FEATURES, ARTIFACTS_SNAPSHOTS,
        FIXED_MODELS, ROLLING_MODELS, OFFER_MODELS,
        FIXED_REPORTS, ROLLING_REPORTS, OFFER_REPORTS,
        ANALYTICS_REPORTS, PLAN_REPORTS, LOG_REPORTS, OUT_OUTPUTS,
    ):
        os.makedirs(path, exist_ok=True)


def read_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, name), low_memory=False, **kwargs)


def to01(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype("int64")
    low = s.astype("string").str.strip().str.lower()
    return low.map({
        "true": 1, "false": 0, "y": 1, "n": 0, "yes": 1, "no": 0,
        "1": 1, "0": 0, "t": 1, "f": 0,
    })


def safe_div(a, b):
    return np.divide(a, b, out=np.full_like(np.asarray(a, dtype=float), np.nan),
                     where=np.asarray(b, dtype=float) != 0)


def sample_ref(ids: Iterable) -> pd.Series:
    def one(value) -> str:
        return "S" + hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:10]

    return pd.Series([one(v) for v in ids], dtype="string")


def offer_catalogue() -> pd.DataFrame:
    offers = read_csv("offer_catalogue.csv")
    offers["requires_promotable"] = to01(offers["requires_promotable_items_clean"]).fillna(0).astype(int)
    offers["counts_monthly"] = pd.to_numeric(
        offers["counts_toward_monthly_limit"], errors="coerce"
    ).fillna(0).astype(int)
    offers["value"] = pd.to_numeric(offers["value"], errors="coerce").fillna(0.0)
    offers["tier_list"] = offers["eligible_tiers"].fillna("All").astype(str).str.split(",")
    return offers[[
        "offer_code", "description", "offer_type", "value", "eligible_tiers",
        "tier_list", "requires_promotable", "counts_monthly",
    ]].copy()


def tiers_for_offer(row) -> list[str]:
    tiers = row["tier_list"]
    if isinstance(tiers, str):
        tiers = [tiers]
    if any(str(t).strip().lower() == "all" for t in tiers):
        return TIERS
    return [str(t).strip() for t in tiers]


def offer_cost_series(df: pd.DataFrame) -> pd.Series:
    aov = pd.to_numeric(df.get("expected_order_value"), errors="coerce").fillna(AOV_FALLBACK)
    value = pd.to_numeric(df.get("offer_value"), errors="coerce").fillna(0.0)
    kind = df.get("offer_type", pd.Series("No offer", index=df.index)).astype(str)
    cost = np.select(
        [kind.eq("Shipping"), kind.eq("Voucher"), kind.eq("Points"), kind.eq("Credit")],
        [6.0, value * aov, 3.0, 10.0], default=0.0,
    )
    return pd.Series(cost, index=df.index, dtype=float)


def read_budget() -> pd.DataFrame:
    budget = read_csv("retention_budget_2024Q4.csv")
    budget["week_start"] = pd.to_datetime(budget["week_start"])
    budget["country"] = budget["country"].astype(str)
    return budget


def region_country_map() -> pd.Series:
    customers = read_csv("customers.csv", usecols=["region", "country_code_clean"])
    out = customers.dropna().groupby("region")["country_code_clean"] \
        .agg(lambda s: s.mode().iloc[0])
    out.index.name = "cust_region"
    return out


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
