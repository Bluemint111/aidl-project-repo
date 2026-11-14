"""Validate the cleaned model input before training or scoring."""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from shopsphere.common import DATA_DIR, OUT_REPORTS, ensure_dirs
from shopsphere.data.asof_time import parse_datetime

REQUIRED = {
    "customers.csv": [
        "customer_id", "age", "country_code_clean", "region", "signup_date",
        "loyalty_score", "email_open_rate", "discount_usage_rate",
        "avg_review_score", "referral_code", "customer_tier", "credit_limit",
    ],
    "customer_preferences.csv": [
        "customer_id", "email_opt_in_clean", "sms_opt_in_clean",
        "preferred_channel", "preferred_language", "do_not_contact_clean",
        "last_updated",
    ],
    "transactions.csv": [
        "transaction_id", "customer_id", "transaction_timestamp", "order_value",
        "items_count", "payment_method", "discount_applied_clean",
        "shipping_speed", "high_value_flag_clean",
    ],
    "transaction_items.csv": [
        "transaction_id", "line_no", "product_id", "quantity", "unit_price",
    ],
    "shopsphere_sessions_cleaned.csv": [
        "session_id", "customer_id", "session_timestamp", "session_duration",
        "pages_viewed", "cart_additions", "campaign_id", "geo_ip_region",
        "traffic_source_clean", "device_type_clean", "bounce_clean",
        "campaign_status",
    ],
    "products.csv": [
        "product_id", "category", "product_name", "unit_price", "margin_pct",
        "stock_on_hand", "next_restock_date", "is_promotable_clean",
    ],
    "deliveries.csv": [
        "transaction_id", "carrier", "shipping_speed", "ship_date",
        "promised_date", "delivered_date", "delay_days", "delivery_status",
    ],
    "support_tickets.csv": [
        "ticket_id", "customer_id", "transaction_id", "created_date",
        "channel_clean", "category", "priority", "status",
        "first_response_hours", "resolution_hours", "csat_score", "message",
    ],
    "retention_contact_history.csv": [
        "contact_id", "customer_id", "contact_date", "channel", "offer_code",
        "purchased_within_14d_clean", "redeemed_clean", "redemption_date",
    ],
    "offer_catalogue.csv": [
        "offer_code", "description", "offer_type", "value", "eligible_tiers",
        "counts_toward_monthly_limit", "requires_promotable_items_clean",
    ],
    "campaign_daily_performance.csv": [
        "campaign_id", "date", "spend", "impressions", "clicks",
        "attributed_sessions", "attributed_orders", "attributed_revenue",
    ],
    "marketing_campaigns.csv": [
        "campaign_id", "campaign_type", "campaign_budget", "region_target",
        "start_date", "end_date", "status",
    ],
    "geo_data.csv": [
        "geo_ip_region", "average_income", "urban_ratio",
        "internet_penetration", "region_tier",
    ],
    "retention_budget_2024Q4.csv": [
        "week_start", "country", "retention_budget_usd",
    ],
    "train.csv": ["customer_id", "target"],
    "test.csv": ["customer_id"],
    "sample_submission.csv": ["customer_id", "target"],
}

KEYS = {
    "customers.csv": ["customer_id"],
    "customer_preferences.csv": ["customer_id"],
    "transactions.csv": ["transaction_id"],
    "transaction_items.csv": ["transaction_id", "line_no"],
    "shopsphere_sessions_cleaned.csv": ["session_id"],
    "products.csv": ["product_id"],
    "deliveries.csv": ["transaction_id"],
    "support_tickets.csv": ["ticket_id"],
    "retention_contact_history.csv": ["contact_id"],
    "offer_catalogue.csv": ["offer_code"],
    "campaign_daily_performance.csv": ["campaign_id", "date"],
    "marketing_campaigns.csv": ["campaign_id"],
    "geo_data.csv": ["geo_ip_region"],
    "retention_budget_2024Q4.csv": ["week_start", "country"],
    "train.csv": ["customer_id"],
    "test.csv": ["customer_id"],
    "sample_submission.csv": ["customer_id"],
}

REFERENCES = [
    ("customer_preferences.csv", "customer_id", "customers.csv", "customer_id"),
    ("transactions.csv", "customer_id", "customers.csv", "customer_id"),
    ("transaction_items.csv", "transaction_id", "transactions.csv", "transaction_id"),
    ("shopsphere_sessions_cleaned.csv", "customer_id", "customers.csv", "customer_id"),
    ("shopsphere_sessions_cleaned.csv", "campaign_id", "marketing_campaigns.csv", "campaign_id"),
    ("deliveries.csv", "transaction_id", "transactions.csv", "transaction_id"),
    ("support_tickets.csv", "customer_id", "customers.csv", "customer_id"),
    ("support_tickets.csv", "transaction_id", "transactions.csv", "transaction_id"),
    ("retention_contact_history.csv", "customer_id", "customers.csv", "customer_id"),
    ("retention_contact_history.csv", "offer_code", "offer_catalogue.csv", "offer_code"),
    ("campaign_daily_performance.csv", "campaign_id", "marketing_campaigns.csv", "campaign_id"),
]

DATE_COLUMNS = {
    "customers.csv": ["signup_date"],
    "customer_preferences.csv": ["last_updated"],
    "transactions.csv": ["transaction_timestamp"],
    "shopsphere_sessions_cleaned.csv": ["session_timestamp"],
    "products.csv": ["next_restock_date"],
    "deliveries.csv": ["ship_date", "promised_date", "delivered_date"],
    "support_tickets.csv": ["created_date"],
    "retention_contact_history.csv": ["contact_date", "redemption_date"],
    "campaign_daily_performance.csv": ["date"],
    "marketing_campaigns.csv": ["start_date", "end_date"],
    "retention_budget_2024Q4.csv": ["week_start"],
}

NONNEGATIVE = {
    "customers.csv": ["age", "loyalty_score", "discount_usage_rate", "credit_limit"],
    "transactions.csv": ["order_value", "items_count"],
    "transaction_items.csv": ["quantity", "unit_price"],
    "products.csv": ["unit_price", "margin_pct", "stock_on_hand"],
    "deliveries.csv": ["delay_days"],
    "support_tickets.csv": ["first_response_hours", "resolution_hours"],
    "campaign_daily_performance.csv": [
        "spend", "impressions", "clicks", "attributed_sessions",
        "attributed_orders", "attributed_revenue",
    ],
    "retention_budget_2024Q4.csv": ["retention_budget_usd"],
}


def _read(data_dir: str, name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, name), low_memory=False)


def validate(data_dir: str = DATA_DIR, strict: bool = False) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    files: dict[str, dict] = {}
    date_ranges: dict[str, dict] = {}

    for name, required in REQUIRED.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            errors.append(f"missing file: {name}")
            continue
        df = _read(data_dir, name)
        frames[name] = df
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(f"{name}: missing columns {missing}")
        key_cols = [c for c in KEYS.get(name, []) if c in df.columns]
        duplicate_count = 0
        if key_cols:
            duplicate_count = int(df.duplicated(key_cols).sum())
            if duplicate_count:
                errors.append(f"{name}: {duplicate_count} duplicate key rows on {key_cols}")
        nulls = {
            c: int(df[c].isna().sum())
            for c in df.columns
            if int(df[c].isna().sum()) > 0
        }
        files[name] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "nulls": nulls,
            "duplicate_key_rows": duplicate_count,
        }

    for child, child_col, parent, parent_col in REFERENCES:
        if child not in frames or parent not in frames:
            continue
        if child_col not in frames[child].columns or parent_col not in frames[parent].columns:
            continue
        children = frames[child][child_col].dropna()
        parents = set(frames[parent][parent_col].dropna())
        orphans = int((~children.isin(parents)).sum())
        if orphans:
            errors.append(f"{child}.{child_col}: {orphans} orphan values")

    for name, columns in DATE_COLUMNS.items():
        if name not in frames:
            continue
        date_ranges[name] = {}
        for col in columns:
            if col not in frames[name].columns:
                continue
            raw = frames[name][col]
            nonblank = raw.notna() & raw.astype(str).str.strip().ne("")
            parsed = parse_datetime(raw, errors="coerce")
            bad = int((nonblank & parsed.isna()).sum())
            if bad:
                warnings.append(f"{name}.{col}: {bad} unparsed dates")
            if parsed.notna().any():
                date_ranges[name][col] = {
                    "min": str(parsed.min().date()),
                    "max": str(parsed.max().date()),
                }

    for name, columns in NONNEGATIVE.items():
        if name not in frames:
            continue
        for col in columns:
            if col not in frames[name].columns:
                continue
            values = pd.to_numeric(frames[name][col], errors="coerce")
            negative = int((values < 0).sum())
            if negative:
                warnings.append(f"{name}.{col}: {negative} negative values")

    for name, target in [("train.csv", "target"), ("sample_submission.csv", "target")]:
        if name in frames and target in frames[name].columns:
            bad = int((~frames[name][target].dropna().isin([0, 1])).sum())
            if bad:
                errors.append(f"{name}.{target}: {bad} values outside 0/1")

    report = {
        "ok": not errors,
        "data_dir": os.path.abspath(data_dir),
        "files": files,
        "date_ranges": date_ranges,
        "errors": errors,
        "warnings": warnings,
    }
    ensure_dirs()
    out_path = os.path.join(OUT_REPORTS, "data_quality.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    report["report_path"] = out_path
    if strict and errors:
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ShopSphere clean data")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = validate(args.data_dir, args.strict)
    print(json.dumps({
        "ok": report["ok"],
        "files": len(report["files"]),
        "errors": report["errors"],
        "warnings": report["warnings"],
        "report_path": report["report_path"],
    }, indent=2))


if __name__ == "__main__":
    main()
