"""Time helpers shared by the rolling/live ShopSphere pipelines.

Nothing here is tied to a specific campaign date.  The as-of date is either
passed in explicitly, read from ``SHOPSPHERE_AS_OF``, or derived from the
latest event timestamp in the data pack.
"""

from __future__ import annotations

import os

import pandas as pd


def parse_datetime(values, **kwargs):
    """Parse date strings that may mix date-only and timestamp formats."""
    try:
        return pd.to_datetime(values, format="mixed", **kwargs)
    except (TypeError, ValueError):
        return pd.to_datetime(values, **kwargs)


def resolve_as_of(
    as_of=None,
    data_dir: str | None = None,
    data_file: str = "transactions.csv",
    column: str = "transaction_timestamp",
) -> pd.Timestamp:
    """Return the as-of date for a snapshot.

    Priority: explicit argument, environment variable, latest event in data.
    """
    if as_of is not None:
        return pd.Timestamp(as_of).normalize()

    for key in ("SHOPSPHERE_AS_OF", "AS_OF"):
        value = os.environ.get(key)
        if value:
            return pd.Timestamp(value).normalize()

    if data_dir is None:
        raise ValueError("as_of, SHOPSPHERE_AS_OF, or data_dir is required")

    path = os.path.join(data_dir, data_file)
    if os.path.exists(path):
        dates = parse_datetime(
            pd.read_csv(path, usecols=[column])[column], errors="coerce"
        )
        latest = dates.max()
        if pd.notna(latest):
            return pd.Timestamp(latest).normalize()

    raise ValueError(f"Could not derive as-of date from {path}")


def window_bounds(as_of, days: int = 30) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the inclusive target window (as_of + 1 day .. as_of + days)."""
    start = pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1)
    end = pd.Timestamp(as_of).normalize() + pd.Timedelta(days=days)
    return start, end


def target_months(as_of, days: int = 30) -> set[int]:
    """Month numbers touched by the target window."""
    start, end = window_bounds(as_of, days)
    return set(pd.date_range(start, end, freq="D").month.tolist())


def month_end_dates(end, count: int) -> list[pd.Timestamp]:
    """Return ``count`` month-end dates ending with the month containing end."""
    cur = (pd.Timestamp(end) + pd.offsets.MonthEnd(0)).normalize()
    dates = []
    for _ in range(int(count)):
        dates.append(cur)
        cur = (cur.replace(day=1) - pd.Timedelta(days=1)).normalize()
    return sorted(dates)


def weekly_dates(end, count: int) -> list[pd.Timestamp]:
    """Return ``count`` weekly dates ending at end."""
    cur = pd.Timestamp(end).normalize()
    dates = [cur - pd.Timedelta(days=7 * i) for i in range(int(count))]
    return sorted(dates)


def black_friday(year: int) -> pd.Timestamp:
    """Black Friday for a year (the day after the fourth Thursday in November)."""
    nov1 = pd.Timestamp(year=int(year), month=11, day=1)
    days_to_thursday = (3 - nov1.weekday()) % 7
    first_thursday = nov1 + pd.Timedelta(days=days_to_thursday)
    fourth_thursday = first_thursday + pd.Timedelta(days=21)
    return fourth_thursday + pd.Timedelta(days=1)


def next_black_friday(start) -> pd.Timestamp:
    start = pd.Timestamp(start).normalize()
    for year in (start.year, start.year + 1):
        candidate = black_friday(year)
        if candidate >= start:
            return candidate
    return black_friday(start.year + 2)


def next_christmas(start) -> pd.Timestamp:
    start = pd.Timestamp(start).normalize()
    candidate = pd.Timestamp(year=start.year, month=12, day=25)
    return candidate if candidate >= start else pd.Timestamp(year=start.year + 1, month=12, day=25)
