from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

from shopsphere.common import (
    CHANNELS, MODEL_FEATURES, OFFER_MODELS as OUT_MODELS,
    OFFER_REPORTS as OUT_REPORTS, OUT_DATA, ensure_dirs, offer_catalogue,
    offer_cost_series, tiers_for_offer,
)

CHUNK = 10000

WINDOW_COLUMNS = [
    "target_window_start", "target_window_end", "target_month_start", "target_month_end",
    "target_month_sin", "target_month_cos", "target_is_holiday", "target_is_q4",
    "target_days_to_black_friday", "target_days_to_christmas",
    "target_is_black_friday_window", "target_is_christmas_window", "target_peak_proximity",
    "country_target_month_index", "region_target_month_index",
]


def channel_ok(df: pd.DataFrame) -> pd.Series:
    ok = pd.Series(True, index=df.index)
    ok &= df["channel"].eq("email") & df["pref_email_opt"].fillna(0).astype(int).eq(1) | \
        df["channel"].ne("email")
    ok &= df["channel"].eq("sms") & df["pref_sms_opt"].fillna(0).astype(int).eq(1) | \
        df["channel"].ne("sms")
    ok &= df["pref_dnc"].fillna(0).astype(int).eq(0)
    return ok


def service_ok(df: pd.DataFrame) -> pd.Series:
    recovery = df["sup_recent_complaint"].fillna(0).astype(int).eq(1) & \
        df["sup_open_n"].fillna(0).gt(0)
    return ~recovery | df["offer_code"].eq("OFF-SRC")


def tier_offer_lookup(offers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, offer in offers.iterrows():
        for tier in tiers_for_offer(offer):
            rows.append((offer["offer_code"], tier))
    return pd.DataFrame(rows, columns=["offer_code", "cust_tier"])


def candidate_frame(cur: pd.DataFrame, offers: pd.DataFrame) -> pd.DataFrame:
    cand = cur.merge(offers, how="cross")
    cand = cand[cand["offer_code"] != "OFF-NONE"].copy()
    lookup = tier_offer_lookup(offers)
    cand = cand.merge(lookup, on=["offer_code", "cust_tier"], how="inner")
    cand = cand[(cand["offer_requires_promotable"] == 0) |
                (cand["item_promotable_share"].fillna(0) > 0)].copy()
    cand = cand.merge(pd.DataFrame({"channel": CHANNELS}), how="cross")
    cand = cand[channel_ok(cand) & service_ok(cand)].copy()
    cand["tier_offer_match"] = 1
    cand["contact_channel_is_preferred"] = (
        cand["channel"].astype(str) == cand["pref_channel"].astype(str)
    ).astype(int)
    cand["offer_face_value"] = offer_cost_series(cand)
    cand["offer_cost"] = cand["offer_face_value"]
    return cand


def baseline_frame(cur: pd.DataFrame, offers: pd.DataFrame) -> pd.DataFrame:
    none_offer = offers[offers["offer_code"] == "OFF-NONE"].iloc[0]
    base = cur.copy()
    for col in ["offer_code", "offer_type", "offer_value", "eligible_tiers"]:
        base[col] = none_offer[col]
    base["offer_requires_promotable"] = int(none_offer["offer_requires_promotable"])
    base["offer_counts_monthly"] = int(none_offer["offer_counts_monthly"])
    base = base.merge(pd.DataFrame({"channel": CHANNELS}), how="cross")
    base = base[channel_ok(base)].copy()
    base["tier_offer_match"] = 1
    base["contact_channel_is_preferred"] = (
        base["channel"].astype(str) == base["pref_channel"].astype(str)
    ).astype(int)
    base["offer_face_value"] = 0.0
    base["offer_cost"] = 0.0
    return base


def main() -> None:
    ensure_dirs()
    cur = pd.read_pickle(os.path.join(OUT_DATA, "current_customers.pkl"))
    offers = offer_catalogue().rename(columns={
        "value": "offer_value", "requires_promotable": "offer_requires_promotable",
        "counts_monthly": "offer_counts_monthly",
    })
    pre = joblib.load(os.path.join(OUT_MODELS, "offer_preprocessor.joblib"))
    purchase = joblib.load(os.path.join(OUT_MODELS, "purchase_model.joblib"))
    redemption = joblib.load(os.path.join(OUT_MODELS, "redemption_model.joblib"))
    with open(os.path.join(OUT_MODELS, "offer_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)

    keep = [
        "customer_id", "sample_ref", "cust_country", "cust_region", "cust_tier",
        "pref_channel", "last_contact_date", "expected_order_value", "margin",
        "w1_probability", "w1_target", "w1_as_of", "w1_window_start", "w1_window_end",
    ] + WINDOW_COLUMNS
    base_cols = list(dict.fromkeys(keep + MODEL_FEATURES + [
        "pref_email_opt", "pref_sms_opt", "pref_dnc", "sup_open_n",
        "sup_recent_complaint", "item_promotable_share",
    ]))
    cur = cur[[c for c in base_cols if c in cur.columns]].copy()

    rows = []
    for start in range(0, len(cur), CHUNK):
        chunk = cur.iloc[start:start + CHUNK].copy()
        cand = candidate_frame(chunk, offers)
        base = baseline_frame(chunk, offers)

        Xc = pre.transform(cand[MODEL_FEATURES]).astype(np.float32)
        cand["purchase_probability"] = purchase.predict_proba(Xc)[:, 1]
        cand["redemption_probability"] = redemption.predict_proba(Xc)[:, 1]
        cand["offer_cost"] = cand["offer_face_value"] * cand["redemption_probability"]

        Xb = pre.transform(base[MODEL_FEATURES]).astype(np.float32)
        base["baseline_probability"] = purchase.predict_proba(Xb)[:, 1]
        cand = cand.merge(
            base[["customer_id", "channel", "baseline_probability"]],
            on=["customer_id", "channel"], how="left",
        )
        cand["uplift"] = cand["purchase_probability"] - cand["baseline_probability"]
        cand["expected_incremental_revenue"] = cand["uplift"] * cand["expected_order_value"]
        cand["expected_incremental_margin"] = (
            cand["expected_incremental_revenue"] * cand["margin"]
        )
        cand["expected_net_value"] = cand["expected_incremental_margin"] - cand["offer_cost"]
        cand["roi"] = np.where(
            cand["offer_cost"] > 0,
            cand["expected_net_value"] / cand["offer_cost"].replace(0, np.nan),
            np.nan,
        )

        out_cols = [
            "sample_ref", "cust_country", "cust_region", "cust_tier", "channel",
            "offer_code", "offer_type", "offer_value", "purchase_probability",
            "baseline_probability", "uplift", "redemption_probability",
            "expected_order_value", "margin", "offer_face_value", "offer_cost",
            "expected_incremental_revenue", "expected_incremental_margin",
            "expected_net_value", "roi", "pref_channel", "w1_probability",
            "w1_target", "w1_as_of", "w1_window_start", "w1_window_end",
            "last_contact_date",
        ] + WINDOW_COLUMNS
        rows.append(cand[out_cols].copy())
        print(f"scored {min(start + CHUNK, len(cur)):,}/{len(cur):,} customers")

    scores = pd.concat(rows, ignore_index=True)
    for col in ["purchase_probability", "baseline_probability", "uplift",
                "redemption_probability", "expected_order_value", "margin", "offer_cost",
                "offer_face_value", "expected_incremental_revenue", "expected_incremental_margin",
                "expected_net_value", "roi", "w1_probability"]:
        scores[col] = pd.to_numeric(scores[col], errors="coerce").astype("float32")
    scores["w1_target"] = scores["w1_target"].fillna(0).astype("int8")
    scores["last_contact_date"] = pd.to_datetime(scores["last_contact_date"], errors="coerce")

    path = os.path.join(OUT_DATA, "offer_candidate_scores.parquet")
    scores.to_parquet(path, index=False)
    summary = {
        "rows": int(len(scores)),
        "customers": int(scores["sample_ref"].nunique()),
        "offers": sorted(scores["offer_code"].unique().tolist()),
        "channels": sorted(scores["channel"].unique().tolist()),
        "countries": sorted(scores["cust_country"].unique().tolist()),
        "positive_uplift_share": round(float((scores["uplift"] > 0).mean()), 5),
        "positive_net_share": round(float((scores["expected_net_value"] > 0).mean()), 5),
        "selected_purchase_model": meta["selected_purchase_model"],
        "score_as_of": str(pd.Timestamp(scores["w1_as_of"].iloc[0]).date())
        if "w1_as_of" in scores else None,
        "score_window_start": str(pd.Timestamp(scores["w1_window_start"].iloc[0]).date())
        if "w1_window_start" in scores else None,
        "score_window_end": str(pd.Timestamp(scores["w1_window_end"].iloc[0]).date())
        if "w1_window_end" in scores else None,
        "target_window_start": str(pd.Timestamp(scores["target_window_start"].iloc[0]).date())
        if "target_window_start" in scores else None,
        "target_is_holiday": int(scores["target_is_holiday"].iloc[0])
        if "target_is_holiday" in scores else None,
        "target_days_to_black_friday": int(scores["target_days_to_black_friday"].iloc[0])
        if "target_days_to_black_friday" in scores else None,
    }
    with open(os.path.join(OUT_REPORTS, "offer_candidate_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
