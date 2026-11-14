from __future__ import annotations

import argparse
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

import pandas as pd

from shopsphere.common import DB_PATH, OUT_OUTPUTS, ensure_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_ref TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL,
    week_start TEXT NOT NULL,
    country TEXT NOT NULL,
    objective TEXT NOT NULL,
    budget_usd REAL,
    planned_contacts INTEGER NOT NULL,
    planned_spend REAL,
    expected_incremental_revenue REAL,
    expected_net_value REAL,
    expected_roi REAL,
    status TEXT NOT NULL DEFAULT 'Pending',
    decided_by TEXT,
    decided_at TEXT,
    decision_reason TEXT,
    summary_json TEXT
);
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL,
    sample_ref TEXT NOT NULL,
    cust_country TEXT,
    cust_tier TEXT,
    channel TEXT,
    offer_code TEXT,
    offer_type TEXT,
    purchase_probability REAL,
    baseline_probability REAL,
    uplift REAL,
    redemption_probability REAL,
    expected_order_value REAL,
    offer_cost REAL,
    expected_incremental_revenue REAL,
    expected_net_value REAL,
    roi REAL,
    w1_probability REAL,
    priority_rank INTEGER,
    FOREIGN KEY (batch_id) REFERENCES batches(id)
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (batch_id) REFERENCES batches(id)
);
CREATE INDEX IF NOT EXISTS idx_recommendations_batch ON recommendations(batch_id);
CREATE INDEX IF NOT EXISTS idx_audit_batch ON audit_log(batch_id);
"""


def connect() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def py(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def create_batch(plan: pd.DataFrame, summary: dict, created_by: str = "system") -> str:
    if plan is None or len(plan) == 0:
        raise ValueError("Cannot create an empty approval batch")
    if not str(created_by).strip():
        raise ValueError("created_by is required")
    batch_ref = (
        f"BATCH-{summary['week_start']}-{summary['country']}-"
        f"{datetime.now().strftime('%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"
    )
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO batches (
                batch_ref, created_at, created_by, week_start, country, objective,
                budget_usd, planned_contacts, planned_spend, expected_incremental_revenue,
                expected_net_value, expected_roi, status, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Pending', ?)""",
            (
                batch_ref, now(), created_by, summary["week_start"], summary["country"],
                summary["objective"], py(summary.get("budget_usd")),
                int(summary["planned_contacts"]), py(summary.get("planned_spend")),
                py(summary.get("expected_incremental_revenue")),
                py(summary.get("expected_net_value")), py(summary.get("expected_roi")),
                json.dumps(summary, default=str),
            ),
        )
        batch_id = int(cur.lastrowid)
        rec_cols = [
            "sample_ref", "cust_country", "cust_tier", "channel", "offer_code",
            "offer_type", "purchase_probability", "baseline_probability", "uplift",
            "redemption_probability", "expected_order_value", "offer_cost",
            "expected_incremental_revenue", "expected_net_value", "roi",
            "w1_probability", "priority_rank",
        ]
        rows = []
        for _, r in plan.iterrows():
            rows.append(tuple([batch_id] + [py(r.get(c)) for c in rec_cols]))
        placeholders = ", ".join(["?"] * (len(rec_cols) + 1))
        cur.executemany(
            f"INSERT INTO recommendations (batch_id, {', '.join(rec_cols)}) "
            f"VALUES ({placeholders})", rows,
        )
        cur.execute(
            "INSERT INTO audit_log (batch_id, action, actor, reason, created_at) "
            "VALUES (?, 'created', ?, ?, ?)",
            (batch_id, created_by, f"Created batch with {len(plan)} recommendations", now()),
        )
        conn.commit()
        return batch_ref
    finally:
        conn.close()


def list_batches() -> pd.DataFrame:
    conn = connect()
    try:
        return pd.read_sql_query(
            "SELECT * FROM batches ORDER BY id DESC", conn,
        )
    finally:
        conn.close()


def get_batch(batch_ref: str) -> tuple[pd.Series, pd.DataFrame]:
    conn = connect()
    try:
        batch = pd.read_sql_query(
            "SELECT * FROM batches WHERE batch_ref = ?", conn, params=(batch_ref,),
        )
        if len(batch) == 0:
            raise KeyError(f"Unknown batch {batch_ref!r}")
        recs = pd.read_sql_query(
            "SELECT * FROM recommendations WHERE batch_id = ? ORDER BY priority_rank",
            conn, params=(int(batch["id"].iloc[0]),),
        )
        return batch.iloc[0], recs
    finally:
        conn.close()


def audit_trail(batch_ref: str) -> pd.DataFrame:
    conn = connect()
    try:
        return pd.read_sql_query(
            """SELECT a.* FROM audit_log a
               JOIN batches b ON b.id = a.batch_id
               WHERE b.batch_ref = ? ORDER BY a.id""",
            conn, params=(batch_ref,),
        )
    finally:
        conn.close()


def decide_batch(batch_ref: str, decision: str, actor: str, reason: str) -> None:
    if decision not in ("Approved", "Rejected"):
        raise ValueError("decision must be Approved or Rejected")
    if not str(actor).strip():
        raise ValueError("actor is required")
    if not str(reason).strip():
        raise ValueError("reason is required")
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, status FROM batches WHERE batch_ref = ?", (batch_ref,))
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"Unknown batch {batch_ref!r}")
        if row["status"] != "Pending":
            raise ValueError(f"Batch is already {row['status']}")
        cur.execute(
            """UPDATE batches SET status = ?, decided_by = ?, decided_at = ?,
               decision_reason = ? WHERE batch_ref = ?""",
            (decision, actor, now(), reason, batch_ref),
        )
        cur.execute(
            "INSERT INTO audit_log (batch_id, action, actor, reason, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (int(row["id"]), decision.lower(), actor, reason, now()),
        )
        conn.commit()
    finally:
        conn.close()


def export_approved(batch_ref: str, path: str | None = None) -> str:
    batch, recs = get_batch(batch_ref)
    if batch["status"] != "Approved":
        raise ValueError("Only approved batches can be exported")
    if path is None:
        ensure_dirs()
        path = os.path.join(OUT_OUTPUTS, f"{batch_ref}_approved.csv")
    out = recs.drop(columns=["id", "batch_id"]).copy()
    out.insert(0, "batch_ref", batch_ref)
    out.insert(1, "week_start", batch["week_start"])
    out.insert(2, "country", batch["country"])
    out.to_csv(path, index=False)
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM batches WHERE batch_ref = ?", (batch_ref,))
        batch_id = int(cur.fetchone()["id"])
        cur.execute(
            "INSERT INTO audit_log (batch_id, action, actor, reason, created_at) "
            "VALUES (?, 'exported', 'system', ?, ?)",
            (batch_id, os.path.basename(path), now()),
        )
        conn.commit()
    finally:
        conn.close()
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["init", "list"])
    args = parser.parse_args()
    if args.action == "init":
        connect().close()
        print("Initialized", DB_PATH)
    else:
        print(list_batches().to_string(index=False))


if __name__ == "__main__":
    main()
