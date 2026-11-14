from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from shopsphere.common import (
    DATA_DIR, OUT_DATA, OUT_OUTPUTS, PLAN_REPORTS as OUT_REPORTS,
    ROLLING_MODELS, ensure_dirs, read_budget, resolve_as_of,
)

OBJECTIVES = {
    "incremental_revenue": "Expected incremental revenue",
    "net_value": "Expected net margin after incentive",
    "roi": "Expected net return per dollar of incentive",
    "propensity": "Purchase probability",
}

WINDOW_FIELDS = (
    "w1_as_of", "w1_window_start", "w1_window_end",
    "target_window_start", "target_window_end", "target_month_start", "target_month_end",
    "target_is_holiday", "target_is_q4", "target_days_to_black_friday",
    "target_days_to_christmas", "target_is_black_friday_window",
    "target_is_christmas_window", "target_peak_proximity",
)


def load_scores() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(OUT_DATA, "offer_candidate_scores.parquet"))


def default_week_start() -> pd.Timestamp:
    budget = read_budget()
    weeks = pd.to_datetime(budget["week_start"]).dropna().sort_values().unique()
    if not len(weeks):
        return pd.Timestamp(resolve_as_of(None, DATA_DIR)) + pd.Timedelta(days=7)
    as_of = pd.Timestamp(resolve_as_of(None, DATA_DIR))
    upcoming = [w for w in weeks if pd.Timestamp(w) >= as_of]
    return pd.Timestamp(upcoming[0] if upcoming else weeks[-1])


def budget_for(week_start, country: str) -> float:
    budget = read_budget()
    week = pd.Timestamp(week_start)
    row = budget[(budget["week_start"] == week) & (budget["country"] == country)]
    if len(row):
        return float(row["retention_budget_usd"].iloc[0])
    nearest = budget[budget["country"] == country]
    if len(nearest):
        nearest = nearest.assign(_gap=(nearest["week_start"] - week).abs())
        return float(nearest.sort_values("_gap")["retention_budget_usd"].iloc[0])
    return float(budget["retention_budget_usd"].mean())


def _scalar(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _date_str(value):
    return str(pd.Timestamp(value).date()) if value is not None else None


def score_window_context(scores: pd.DataFrame) -> dict:
    """Read the rolling W1 as-of/target-window context from the score table."""
    if scores is None or scores.empty:
        return {}
    out = {}
    for col in WINDOW_FIELDS:
        if col not in scores.columns:
            continue
        values = scores[col].dropna()
        if values.empty:
            continue
        value = _scalar(values.iloc[0])
        if value is not None:
            out[col] = value
    return out


def _window_month_factor(scores: pd.DataFrame, country: str) -> float | None:
    col = "country_target_month_index"
    if scores is None or col not in scores.columns:
        return None
    rows = scores[scores["cust_country"] == country]
    values = pd.to_numeric(rows[col], errors="coerce").dropna()
    if values.empty:
        values = pd.to_numeric(scores[col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def _month_seasonality_factor(week_start, country: str, horizon_days: int):
    path = os.path.join(OUT_DATA, "analytics_seasonality_month.csv")
    week = pd.Timestamp(week_start)
    end = week + pd.Timedelta(days=horizon_days)
    if not os.path.exists(path):
        return 1.0, "seasonality unavailable; using 1.00x", []
    season = pd.read_csv(path)
    months = sorted({d.month for d in pd.date_range(week, end, freq="D")})
    rows = season[
        (season["country"] == country) & (season["month_num"].isin(months))
    ]
    values = rows["revenue_index_vs_year_mean"].dropna()
    if values.empty:
        return 1.0, "no same-month history; using 1.00x", months
    factor = float(values.median())
    factor = max(0.75, min(1.35, factor))
    names = ", ".join(pd.Timestamp(week.year, m, 1).strftime("%b") for m in months)
    return factor, f"{names} seasonal demand index {factor:.2f}x (week to week+6)", months


def seasonality_factor(
    week_start, country: str, scores: pd.DataFrame | None = None, horizon_days: int = 6
) -> tuple[float, str]:
    """Seasonal demand multiplier for a planning week.

    Prefers the rolling W1 target-window index when the planning week overlaps
    the live score window, and falls back to the same-month historical index.
    """
    week = pd.Timestamp(week_start)
    end = week + pd.Timedelta(days=horizon_days)
    month_factor, month_note, _ = _month_seasonality_factor(
        week, country, horizon_days
    )
    context = score_window_context(scores)
    window_start = context.get("w1_window_start") or context.get("target_window_start")
    window_end = context.get("w1_window_end") or context.get("target_window_end")
    if window_start is not None and window_end is not None:
        window_start = pd.Timestamp(window_start)
        window_end = pd.Timestamp(window_end)
        if week <= window_end and end >= window_start:
            window_factor = _window_month_factor(scores, country)
            if window_factor is not None:
                factor = max(0.75, min(1.35, window_factor))
                note = (
                    f"W1 target window {window_start.date()} to {window_end.date()} "
                    f"seasonal demand index {factor:.2f}x (score as-of "
                    f"{pd.Timestamp(context.get('w1_as_of') or window_start - pd.Timedelta(days=1)).date()})"
                )
                if abs(factor - month_factor) >= 0.01:
                    note += f"; same-month index {month_factor:.2f}x"
                return factor, note
    return month_factor, month_note


def build_plan(
    week_start,
    country: str,
    budget_usd: float | None = None,
    objective: str = "incremental_revenue",
    min_uplift: float = 0.0,
    min_net_value: float | None = 0.0,
    max_contacts: int | None = None,
    cooldown_days: int = 14,
    scores: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective {objective!r}")
    if scores is None:
        scores = load_scores()
    week = pd.Timestamp(week_start)
    df = scores[scores["cust_country"] == country].copy()
    window_context = score_window_context(scores)
    seasonal_factor, seasonal_note = seasonality_factor(week, country, scores)
    df["seasonal_factor"] = seasonal_factor
    df["expected_incremental_revenue"] *= seasonal_factor
    df["expected_incremental_margin"] *= seasonal_factor
    df["expected_net_value"] = df["expected_incremental_margin"] - df["offer_cost"]
    df["roi"] = (
        df["expected_net_value"] / df["offer_cost"].replace(0, pd.NA)
    )
    df["days_since_last_contact"] = (
        (week - pd.to_datetime(df["last_contact_date"])).dt.days.fillna(9999)
    )
    df = df[df["days_since_last_contact"] >= cooldown_days].copy()
    df = df[df["uplift"] >= min_uplift].copy()
    df = df[df["expected_incremental_revenue"] > 0].copy()
    if min_net_value is not None:
        df = df[df["expected_net_value"] >= min_net_value].copy()
    if objective == "roi":
        df = df[(df["expected_net_value"] > 0) & (df["offer_cost"] > 0)].copy()
    elif objective == "net_value":
        df = df[df["expected_net_value"] > 0].copy()

    if objective in ("incremental_revenue", "propensity"):
        df["priority_score"] = df["expected_incremental_revenue"]
    elif objective == "net_value":
        df["priority_score"] = df["expected_net_value"]
    else:
        df["priority_score"] = df["roi"]

    df = df.sort_values(["priority_score", "expected_net_value"], ascending=False)
    df = df.drop_duplicates("sample_ref", keep="first").copy()
    df = df.sort_values(["priority_score", "expected_net_value"], ascending=False)

    budget = float(budget_usd) if budget_usd is not None else budget_for(week, country)
    cumulative = df["offer_cost"].cumsum()
    df = df[cumulative <= budget].copy()
    if max_contacts is not None:
        df = df.head(int(max_contacts)).copy()

    df.insert(0, "week_start", week.date().isoformat())
    df.insert(1, "country", country)
    df["priority_rank"] = range(1, len(df) + 1)

    spend = float(df["offer_cost"].sum())
    inc_rev = float(df["expected_incremental_revenue"].sum())
    inc_margin = float(df["expected_incremental_margin"].sum())
    net = float(df["expected_net_value"].sum())
    score_as_of = scores["w1_as_of"].iloc[0] if "w1_as_of" in scores else None
    score_window_start = scores["w1_window_start"].iloc[0] if "w1_window_start" in scores else None
    score_window_end = scores["w1_window_end"].iloc[0] if "w1_window_end" in scores else None
    rolling_meta_path = os.path.join(ROLLING_MODELS, "rolling_meta.json")
    score_model = None
    if os.path.exists(rolling_meta_path):
        with open(rolling_meta_path, encoding="utf-8") as fh:
            score_model = json.load(fh).get("selected_model")
    summary = {
        "week_start": week.date().isoformat(),
        "week_end": (week + pd.Timedelta(days=6)).date().isoformat(),
        "country": country,
        "objective": objective,
        "score_as_of": str(pd.Timestamp(score_as_of).date()) if score_as_of is not None else None,
        "score_window_start": str(pd.Timestamp(score_window_start).date()) if score_window_start is not None else None,
        "score_window_end": str(pd.Timestamp(score_window_end).date()) if score_window_end is not None else None,
        "score_model": score_model,
        "target_window_start": _date_str(window_context.get("target_window_start")),
        "target_window_end": _date_str(window_context.get("target_window_end")),
        "target_month_start": _scalar(window_context.get("target_month_start")),
        "target_month_end": _scalar(window_context.get("target_month_end")),
        "target_is_holiday": _scalar(window_context.get("target_is_holiday")),
        "target_days_to_black_friday": _scalar(window_context.get("target_days_to_black_friday")),
        "target_days_to_christmas": _scalar(window_context.get("target_days_to_christmas")),
        "objective_label": OBJECTIVES[objective],
        "budget_usd": round(budget, 2),
        "seasonal_factor": round(float(seasonal_factor), 4),
        "seasonal_note": seasonal_note,
        "planned_contacts": int(len(df)),
        "planned_spend": round(spend, 2),
        "budget_used_pct": round(spend / budget, 4) if budget else None,
        "expected_incremental_revenue": round(inc_rev, 2),
        "expected_incremental_margin": round(inc_margin, 2),
        "expected_net_value": round(net, 2),
        "expected_roi": round(net / spend, 4) if spend else None,
        "avg_uplift": round(float(df["uplift"].mean()), 5) if len(df) else None,
        "avg_purchase_probability": round(float(df["purchase_probability"].mean()), 5) if len(df) else None,
        "avg_w1_probability": round(float(df["w1_probability"].mean()), 5) if len(df) else None,
        "offer_mix": df["offer_code"].value_counts().to_dict() if len(df) else {},
        "channel_mix": df["channel"].value_counts().to_dict() if len(df) else {},
        "tier_mix": df["cust_tier"].value_counts().to_dict() if len(df) else {},
    }
    return df, summary


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", default=None,
                        help="Planning week (default: first budget week on/after latest data date)")
    parser.add_argument("--country", default="US")
    parser.add_argument("--objective", default="incremental_revenue", choices=sorted(OBJECTIVES))
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--min-uplift", type=float, default=0.0)
    parser.add_argument("--max-contacts", type=int, default=None)
    parser.add_argument("--cooldown-days", type=int, default=14)
    args = parser.parse_args()
    week = pd.Timestamp(args.week) if args.week else default_week_start()

    plan, summary = build_plan(
        week, args.country, args.budget, args.objective, args.min_uplift,
        max_contacts=args.max_contacts, cooldown_days=args.cooldown_days,
    )
    stem = f"plan_{week.date().isoformat()}_{args.country}_{args.objective}"
    plan.to_csv(os.path.join(OUT_OUTPUTS, stem + ".csv"), index=False)
    with open(os.path.join(OUT_REPORTS, stem + ".json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(plan.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
