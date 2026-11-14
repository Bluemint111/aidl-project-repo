from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED_FILES = (
    "customers.csv",
    "customer_preferences.csv",
    "geo_data.csv",
    "transactions.csv",
    "deliveries.csv",
    "shopsphere_sessions_cleaned.csv",
    "support_tickets.csv",
    "retention_contact_history.csv",
    "train.csv",
    "test.csv",
)


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def _money_to_number(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.replace(r"[$,]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _add_windowed_aggregates(
    features: pd.DataFrame,
    events: pd.DataFrame,
    *,
    date_column: str,
    cutoff: pd.Timestamp,
    name: str,
    aggregations: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    end = cutoff.normalize() + pd.Timedelta(days=1)
    in_range = events.loc[events[date_column].lt(end)]
    for window, days in (("30d", 30), ("90d", 90), ("365d", 365)):
        start = cutoff.normalize() - pd.Timedelta(days=days - 1)
        recent = in_range.loc[in_range[date_column].ge(start)]
        if recent.empty:
            continue
        summary = recent.groupby("customer_id").agg(**aggregations)
        summary.columns = [f"{name}_{column}_{window}" for column in summary.columns]
        features = features.join(summary, how="left")
    return features


def build_feature_table(data_dir: str | Path) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Build leakage-conscious features from history available at the data cutoff."""
    data_path = Path(data_dir)
    missing = [name for name in REQUIRED_FILES if not (data_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required data files: " + ", ".join(missing)
        )

    customers = _read_csv(
        data_path / "customers.csv", dtype={"customer_id": "string"}
    )
    preferences = _read_csv(
        data_path / "customer_preferences.csv", dtype={"customer_id": "string"}
    )
    geo = _read_csv(data_path / "geo_data.csv", dtype={"geo_ip_region": "string"})
    transactions = _read_csv(
        data_path / "transactions.csv",
        dtype={"customer_id": "string", "transaction_id": "string"},
        usecols=[
            "transaction_id",
            "customer_id",
            "transaction_timestamp",
            "order_value",
            "items_count",
            "discount_applied",
        ],
    )
    sessions = _read_csv(
        data_path / "shopsphere_sessions_cleaned.csv",
        dtype={"customer_id": "string"},
        usecols=[
            "customer_id",
            "session_timestamp",
            "session_duration",
            "pages_viewed",
            "cart_additions",
            "bounce_clean",
        ],
    )
    tickets = _read_csv(
        data_path / "support_tickets.csv",
        dtype={"customer_id": "string"},
        usecols=[
            "customer_id",
            "created_date",
            "priority",
            "status",
            "resolution_hours",
            "csat_score",
        ],
    )
    contacts = _read_csv(
        data_path / "retention_contact_history.csv",
        dtype={"customer_id": "string"},
        usecols=[
            "customer_id",
            "contact_date",
            "purchased_within_14d",
            "redeemed",
        ],
    )
    deliveries = _read_csv(
        data_path / "deliveries.csv",
        dtype={"transaction_id": "string"},
        usecols=["transaction_id", "delay_days", "delivery_status"],
    )

    date_fields = [
        (transactions, "transaction_timestamp"),
        (sessions, "session_timestamp"),
        (tickets, "created_date"),
        (contacts, "contact_date"),
    ]
    latest_dates: list[pd.Timestamp] = []
    for frame, column in date_fields:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
        latest = frame[column].max()
        if pd.notna(latest):
            latest_dates.append(latest.normalize())
    if not latest_dates:
        raise ValueError("No valid activity dates were found in the history files.")
    cutoff = max(latest_dates)

    customer_columns = [
        "customer_id",
        "age",
        "country_code",
        "region",
        "signup_date",
        "loyalty_score",
        "email_open_rate",
        "discount_usage_rate",
        "avg_review_score",
        "customer_tier",
        "credit_limit",
    ]
    absent = [column for column in customer_columns if column not in customers]
    if absent:
        raise ValueError("customers.csv is missing columns: " + ", ".join(absent))
    features = customers[customer_columns].copy().set_index("customer_id")
    features["credit_limit"] = _money_to_number(features["credit_limit"])
    features["signup_date"] = pd.to_datetime(features["signup_date"], errors="coerce")
    features["account_age_days"] = (
        cutoff - features["signup_date"].dt.normalize()
    ).dt.days.clip(lower=0)
    features = features.drop(columns="signup_date")

    preference_columns = [
        "customer_id",
        "email_opt_in",
        "sms_opt_in",
        "preferred_channel",
        "preferred_language",
        "do_not_contact",
    ]
    preferences = preferences[preference_columns]
    features = features.join(preferences.set_index("customer_id"), how="left")
    features = features.join(
        geo.set_index("geo_ip_region"),
        on="region",
        how="left",
    )

    transactions["order_value_num"] = _money_to_number(transactions["order_value"])
    transactions["items_count"] = pd.to_numeric(
        transactions["items_count"], errors="coerce"
    )
    transactions["discount_flag"] = (
        transactions["discount_applied"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin({"yes", "true", "1", "y"})
        .astype(float)
    )
    transactions = transactions.loc[
        transactions["transaction_timestamp"].lt(cutoff + pd.Timedelta(days=1))
    ].copy()
    transaction_summary = transactions.groupby("customer_id").agg(
        transaction_count=("transaction_id", "count"),
        revenue_total=("order_value_num", "sum"),
        order_value_mean=("order_value_num", "mean"),
        items_per_order_mean=("items_count", "mean"),
        discount_rate=("discount_flag", "mean"),
        last_purchase=("transaction_timestamp", "max"),
    )
    features = features.join(transaction_summary, how="left")
    features["days_since_last_purchase"] = (
        cutoff - features.pop("last_purchase").dt.normalize()
    ).dt.days.clip(lower=0)
    features = _add_windowed_aggregates(
        features,
        transactions,
        date_column="transaction_timestamp",
        cutoff=cutoff,
        name="transaction",
        aggregations={
            "count": ("transaction_id", "count"),
            "revenue": ("order_value_num", "sum"),
            "average_order_value": ("order_value_num", "mean"),
            "discount_rate": ("discount_flag", "mean"),
        },
    )

    sessions["session_duration"] = pd.to_numeric(
        sessions["session_duration"], errors="coerce"
    )
    sessions["pages_viewed"] = pd.to_numeric(
        sessions["pages_viewed"], errors="coerce"
    )
    sessions["cart_additions"] = pd.to_numeric(
        sessions["cart_additions"], errors="coerce"
    )
    sessions["bounce_flag"] = (
        sessions["bounce_clean"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin({"true", "yes", "1", "y"})
        .astype(float)
    )
    sessions = sessions.loc[
        sessions["session_timestamp"].lt(cutoff + pd.Timedelta(days=1))
    ].copy()
    session_summary = sessions.groupby("customer_id").agg(
        session_count=("session_timestamp", "count"),
        session_duration_mean=("session_duration", "mean"),
        pages_viewed_mean=("pages_viewed", "mean"),
        cart_additions_total=("cart_additions", "sum"),
        bounce_rate=("bounce_flag", "mean"),
        last_session=("session_timestamp", "max"),
    )
    features = features.join(session_summary, how="left")
    features["days_since_last_session"] = (
        cutoff - features.pop("last_session").dt.normalize()
    ).dt.days.clip(lower=0)
    features = _add_windowed_aggregates(
        features,
        sessions,
        date_column="session_timestamp",
        cutoff=cutoff,
        name="session",
        aggregations={
            "count": ("session_timestamp", "count"),
            "cart_additions": ("cart_additions", "sum"),
            "pages_viewed_mean": ("pages_viewed", "mean"),
        },
    )

    tickets["resolution_hours"] = pd.to_numeric(
        tickets["resolution_hours"], errors="coerce"
    )
    tickets["csat_score"] = pd.to_numeric(tickets["csat_score"], errors="coerce")
    tickets["high_priority_flag"] = (
        tickets["priority"].astype("string").str.casefold().eq("high").astype(float)
    )
    tickets["open_ticket_flag"] = (
        tickets["status"].astype("string").str.casefold().eq("open").astype(float)
    )
    tickets = tickets.loc[tickets["created_date"].lt(cutoff + pd.Timedelta(days=1))]
    ticket_summary = tickets.groupby("customer_id").agg(
        ticket_count=("created_date", "count"),
        high_priority_ticket_count=("high_priority_flag", "sum"),
        open_ticket_count=("open_ticket_flag", "sum"),
        resolution_hours_mean=("resolution_hours", "mean"),
        csat_score_mean=("csat_score", "mean"),
    )
    features = features.join(ticket_summary, how="left")
    features = _add_windowed_aggregates(
        features,
        tickets,
        date_column="created_date",
        cutoff=cutoff,
        name="ticket",
        aggregations={
            "count": ("created_date", "count"),
            "high_priority_count": ("high_priority_flag", "sum"),
        },
    )

    contacts["prior_purchase_after_contact"] = (
        contacts["purchased_within_14d"]
        .astype("string")
        .str.strip()
        .str.upper()
        .eq("Y")
        .astype(float)
    )
    contacts["prior_redemption"] = (
        contacts["redeemed"]
        .astype("string")
        .str.strip()
        .str.upper()
        .eq("Y")
        .astype(float)
    )
    contacts = contacts.loc[contacts["contact_date"].lt(cutoff + pd.Timedelta(days=1))]
    contact_summary = contacts.groupby("customer_id").agg(
        contact_count=("contact_date", "count"),
        prior_contact_purchase_rate=("prior_purchase_after_contact", "mean"),
        prior_redemption_rate=("prior_redemption", "mean"),
        last_contact=("contact_date", "max"),
    )
    features = features.join(contact_summary, how="left")
    features["days_since_last_contact"] = (
        cutoff - features.pop("last_contact").dt.normalize()
    ).dt.days.clip(lower=0)
    features = _add_windowed_aggregates(
        features,
        contacts,
        date_column="contact_date",
        cutoff=cutoff,
        name="contact",
        aggregations={
            "count": ("contact_date", "count"),
            "prior_purchase_rate": ("prior_purchase_after_contact", "mean"),
        },
    )

    delivery_events = transactions[
        ["transaction_id", "customer_id", "transaction_timestamp"]
    ].merge(deliveries, on="transaction_id", how="inner")
    delivery_events["delay_days"] = pd.to_numeric(
        delivery_events["delay_days"], errors="coerce"
    )
    delivery_events["delayed_flag"] = (
        delivery_events["delay_days"].gt(0).astype(float)
    )
    delivery_summary = delivery_events.groupby("customer_id").agg(
        delivery_count=("transaction_id", "count"),
        average_delivery_delay_days=("delay_days", "mean"),
        delayed_delivery_rate=("delayed_flag", "mean"),
    )
    features = features.join(delivery_summary, how="left")

    features = features.replace([np.inf, -np.inf], np.nan)
    return features, cutoff


def train_and_score(
    features: pd.DataFrame,
    labels: pd.Series,
    score_customer_ids: pd.Series,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Fit a supervised MLP and return validation metrics plus ranked scores."""
    labels = labels.copy()
    labels.index = labels.index.astype("string")
    y = pd.to_numeric(labels, errors="raise").astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("Training labels must contain both binary classes 0 and 1.")
    missing_train = y.index.difference(features.index)
    if not missing_train.empty:
        raise ValueError(f"{len(missing_train)} training customers have no features.")
    score_ids = score_customer_ids.astype("string")
    missing_score = score_ids[~score_ids.isin(features.index)]
    if not missing_score.empty:
        raise ValueError(f"{len(missing_score)} scoring customers have no features.")

    x = features.loc[y.index].copy()
    numeric_columns = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in x.columns if column not in numeric_columns]
    x[categorical_columns] = x[categorical_columns].astype("object")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=10),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )
    classifier = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=60,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=6,
        random_state=42,
    )
    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    model.fit(x_train, y_train)

    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    top_count = max(1, int(np.ceil(len(y_validation) * 0.1)))
    top_indices = np.argsort(validation_probabilities)[-top_count:]
    metrics: dict[str, float | int] = {
        "validation_roc_auc": float(
            roc_auc_score(y_validation, validation_probabilities)
        ),
        "validation_average_precision": float(
            average_precision_score(y_validation, validation_probabilities)
        ),
        "validation_positive_rate": float(y_validation.mean()),
        "precision_at_top_10_percent": float(y_validation.iloc[top_indices].mean()),
        "recall_at_top_10_percent": float(
            y_validation.iloc[top_indices].sum() / y_validation.sum()
        ),
        "training_iterations": int(model.named_steps["classifier"].n_iter_),
    }

    model.fit(x, y)
    scoring_features = features.loc[score_ids].copy()
    scoring_features[categorical_columns] = scoring_features[
        categorical_columns
    ].astype("object")
    probabilities = model.predict_proba(scoring_features)[:, 1]
    scores = pd.DataFrame(
        {
            "customer_id": score_ids.to_numpy(),
            "purchase_probability_30d": probabilities,
        }
    ).sort_values("purchase_probability_30d", ascending=False, ignore_index=True)
    scores.insert(0, "rank", np.arange(1, len(scores) + 1))
    return scores, metrics
