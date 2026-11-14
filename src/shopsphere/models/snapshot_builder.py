"""Build as-of snapshots for the rolling ShopSphere propensity model.

The original checkpoint uses one fixed cutoff.  This builder creates a set of
labelled historical snapshots plus one unlabelled "live" snapshot from the
latest event date.  All dates come from the data or CLI/env, never constants.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

from shopsphere.common import ARTIFACTS_SNAPSHOTS, DATA_PROCESSED
from shopsphere.data import asof_features as af
from shopsphere.data.asof_time import resolve_as_of, window_bounds

DEFAULT_OUT = ARTIFACTS_SNAPSHOTS
DEFAULT_LIVE = os.path.join(DATA_PROCESSED, "live_features.pkl")


def build(args) -> dict:
    raw = af.load_raw(args.data_dir)
    latest = resolve_as_of(args.as_of, args.data_dir)
    live_only = bool(getattr(args, "live_only", False))
    dates = [] if live_only else af.snapshot_dates(
        raw, as_of=latest, count=args.count, freq=args.freq, labelled=True
    )
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "latest_data_date": str(latest.date()),
        "window_days": af.WINDOW_DAYS,
        "freq": args.freq,
        "snapshots": [],
    }

    for as_of in dates:
        path = os.path.join(args.out_dir, f"snapshot_{as_of.date().isoformat()}.pkl")
        if os.path.exists(path) and not args.overwrite:
            df = pd.read_pickle(path)
        else:
            df = af.build_snapshot(raw, as_of, args.data_dir, labelled=True)
            df.to_pickle(path)
        start, end = window_bounds(as_of, af.WINDOW_DAYS)
        manifest["snapshots"].append({
            "as_of": str(as_of.date()),
            "window_start": str(start.date()),
            "window_end": str(end.date()),
            "rows": int(len(df)),
            "positive_rate": round(float(df["target"].mean()), 5),
            "path": path,
        })
        print(f"snapshot {as_of.date()} rows={len(df):,} positive={df['target'].mean():.3%}")

    live_as_of = latest
    live = af.build_snapshot(raw, live_as_of, args.data_dir, labelled=False)
    live_path = args.live_out
    os.makedirs(os.path.dirname(live_path), exist_ok=True)
    live.to_pickle(live_path)
    start, end = window_bounds(live_as_of, af.WINDOW_DAYS)
    manifest["live"] = {
        "as_of": str(live_as_of.date()),
        "window_start": str(start.date()),
        "window_end": str(end.date()),
        "rows": int(len(live)),
        "path": live_path,
    }
    print(f"live snapshot {live_as_of.date()} rows={len(live):,} -> {live_path}")

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"manifest -> {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rolling ShopSphere snapshots")
    parser.add_argument("--as-of", default=None,
                        help="Latest data date (default: derive from data or SHOPSPHERE_AS_OF)")
    parser.add_argument("--data-dir", default=af.DATA)
    parser.add_argument("--count", type=int, default=12,
                        help="Number of labelled historical snapshots")
    parser.add_argument("--freq", choices=["monthly", "weekly"], default="monthly")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--live-out", default=DEFAULT_LIVE)
    parser.add_argument("--live-only", action="store_true",
                        help="Build only the unlabelled live snapshot")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
