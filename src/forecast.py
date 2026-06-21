"""Exploratory baseline forecasting for next-day oblast-level alert risk.

This module estimates next-day risk from historical daily features. It is a
simple, explainable baseline for analysis only, not an operational prediction
system.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAILY_TIMESERIES_PATH = PROJECT_ROOT / "data" / "processed" / "daily_oblast_timeseries.csv"
EVENT_CALENDAR_PATH = PROJECT_ROOT / "data" / "processed" / "expanded_event_calendar.csv"
FORECAST_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_daily_oblast.csv"
FORECAST_METRICS_PATH = PROJECT_ROOT / "outputs" / "tables" / "forecast_metrics.json"

TARGET_COLUMN = "had_alert_binary_next_day"
MODEL_NAME = "logistic_regression"
DEFAULT_TEST_SIZE = 0.2
RISK_THRESHOLD = 0.5

CATEGORICAL_FEATURES = ["oblast_name", "event_category"]
NUMERIC_FEATURES = [
    "day_of_week",
    "month",
    "alert_count_lag_1",
    "alert_count_lag_3_mean",
    "alert_count_lag_7_mean",
    "total_minutes_lag_1",
    "total_minutes_lag_7_mean",
    "rolling_alert_probability_7d",
    "is_event_window",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class ForecastResult:
    """Forecast output payload for CLI and downstream reuse."""

    predictions: pd.DataFrame
    metrics: dict[str, Any]
    output_path: Path
    metrics_path: Path


def load_daily_timeseries(path: Path = DAILY_TIMESERIES_PATH) -> pd.DataFrame:
    """Load daily oblast time series."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing daily time series: {path}. Run src/preprocess_timeseries.py first."
        )

    dataframe = pd.read_csv(path)
    required = {
        "date",
        "oblast_name",
        "alert_count",
        "total_alert_minutes",
        "had_alert_binary",
    }
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Daily time series missing required columns: {missing}")

    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    for column in ["alert_count", "total_alert_minutes", "had_alert_binary"]:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    return dataframe.dropna(subset=["date", "oblast_name"])


def load_event_calendar(path: Path = EVENT_CALENDAR_PATH) -> pd.DataFrame | None:
    """Load optional expanded event calendar."""

    if not path.exists():
        return None

    dataframe = pd.read_csv(path)
    required = {"window_start_date", "window_end_date", "category"}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Event calendar missing required columns: {missing}")

    dataframe["window_start_date"] = pd.to_datetime(
        dataframe["window_start_date"],
        errors="coerce",
    )
    dataframe["window_end_date"] = pd.to_datetime(
        dataframe["window_end_date"],
        errors="coerce",
    )
    return dataframe.dropna(subset=["window_start_date", "window_end_date"])


def complete_daily_grid(daily: pd.DataFrame, *, include_future_day: bool = True) -> pd.DataFrame:
    """Build a complete date-by-oblast grid and fill missing dates with zeros.

    The preprocessing table only stores days with alerts. Binary classification
    needs explicit no-alert days, so missing oblast-date rows are filled with
    zero counts before feature engineering.
    """

    min_date = daily["date"].min()
    max_date = daily["date"].max()
    end_date = max_date + pd.Timedelta(days=1) if include_future_day else max_date

    oblasts = sorted(daily["oblast_name"].dropna().unique())
    dates = pd.date_range(min_date, end_date, freq="D")
    grid = pd.MultiIndex.from_product(
        [oblasts, dates],
        names=["oblast_name", "forecast_date"],
    ).to_frame(index=False)

    source = daily.rename(columns={"date": "forecast_date"})
    merged = grid.merge(
        source[
            [
                "forecast_date",
                "oblast_name",
                "alert_count",
                "total_alert_minutes",
                "had_alert_binary",
            ]
        ],
        on=["forecast_date", "oblast_name"],
        how="left",
    )

    is_future = merged["forecast_date"] > max_date
    for column in ["alert_count", "total_alert_minutes", "had_alert_binary"]:
        merged[column] = merged[column].fillna(0)
        merged.loc[is_future, column] = pd.NA

    merged["is_future_forecast"] = is_future
    return merged.sort_values(["oblast_name", "forecast_date"]).reset_index(drop=True)


def add_event_features(
    features: pd.DataFrame,
    event_calendar: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add known calendar event-window features by forecast date."""

    enriched = features.copy()
    enriched["is_event_window"] = 0
    enriched["event_category"] = "none"

    if event_calendar is None or event_calendar.empty:
        return enriched

    date_categories = build_event_window_lookup(event_calendar)
    event_features = pd.DataFrame(
        {
            "forecast_date": list(date_categories.keys()),
            "event_categories": list(date_categories.values()),
        }
    )
    if event_features.empty:
        return enriched

    event_features["event_category"] = event_features["event_categories"].map(
        collapse_event_categories
    )
    event_features["is_event_window"] = 1
    enriched = enriched.merge(
        event_features[["forecast_date", "is_event_window", "event_category"]],
        on="forecast_date",
        how="left",
        suffixes=("", "_event"),
    )
    enriched["is_event_window"] = (
        enriched["is_event_window_event"].fillna(enriched["is_event_window"]).astype(int)
    )
    enriched["event_category"] = enriched["event_category_event"].fillna(
        enriched["event_category"]
    )
    return enriched.drop(columns=["is_event_window_event", "event_category_event"])


def build_event_window_lookup(event_calendar: pd.DataFrame) -> dict[pd.Timestamp, list[str]]:
    """Map each date in an event window to the event categories active on that date."""

    lookup: dict[pd.Timestamp, set[str]] = {}
    for row in event_calendar.itertuples(index=False):
        start = pd.Timestamp(row.window_start_date).normalize()
        end = pd.Timestamp(row.window_end_date).normalize()
        for day in pd.date_range(start, end, freq="D"):
            lookup.setdefault(day, set()).add(str(row.category))
    return {day: sorted(categories) for day, categories in lookup.items()}


def collapse_event_categories(categories: list[str]) -> str:
    if not categories:
        return "none"
    if len(categories) == 1:
        return categories[0]
    return "multiple"


def build_features(
    daily: pd.DataFrame,
    event_calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build no-leakage next-day forecast features.

    Each row is indexed by forecast_date. The target is whether an alert happened
    on that forecast_date, named had_alert_binary_next_day because the forecast
    would be made at the end of observation_date = forecast_date - 1 day.

    Leakage guard: all lag and rolling features are shifted by one day within
    each oblast before rolling statistics are calculated. The current
    forecast_date's alert outcome is never used as its own feature.
    """

    complete = complete_daily_grid(daily, include_future_day=True)
    complete["observation_date"] = complete["forecast_date"] - pd.Timedelta(days=1)
    complete["day_of_week"] = complete["forecast_date"].dt.dayofweek
    complete["month"] = complete["forecast_date"].dt.month

    grouped = complete.groupby("oblast_name", group_keys=False)
    shifted_alert_count = grouped["alert_count"].shift(1)
    shifted_total_minutes = grouped["total_alert_minutes"].shift(1)
    shifted_had_alert = grouped["had_alert_binary"].shift(1)

    complete["alert_count_lag_1"] = shifted_alert_count
    complete["alert_count_lag_3_mean"] = shifted_alert_count.groupby(
        complete["oblast_name"]
    ).transform(lambda series: series.rolling(window=3, min_periods=1).mean())
    complete["alert_count_lag_7_mean"] = shifted_alert_count.groupby(
        complete["oblast_name"]
    ).transform(lambda series: series.rolling(window=7, min_periods=1).mean())
    complete["total_minutes_lag_1"] = shifted_total_minutes
    complete["total_minutes_lag_7_mean"] = shifted_total_minutes.groupby(
        complete["oblast_name"]
    ).transform(lambda series: series.rolling(window=7, min_periods=1).mean())
    complete["rolling_alert_probability_7d"] = shifted_had_alert.groupby(
        complete["oblast_name"]
    ).transform(lambda series: series.rolling(window=7, min_periods=1).mean())

    # The future row has no known outcome; historical rows use the observed
    # outcome for forecast_date as the next-day target from observation_date.
    complete[TARGET_COLUMN] = complete["had_alert_binary"].where(
        ~complete["is_future_forecast"],
        pd.NA,
    )
    complete = add_event_features(complete, event_calendar)

    # Remove rows without at least one prior day of history. This avoids filling
    # lag_1 from future or same-day data.
    complete = complete[complete["alert_count_lag_1"].notna()].copy()
    complete["event_category"] = complete["event_category"].fillna("none")
    complete["is_event_window"] = complete["is_event_window"].fillna(0).astype(int)
    return complete


def split_train_test(
    features: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Time-based split using forecast_date, never a random split."""

    known = features[features[TARGET_COLUMN].notna()].copy()
    future = features[features[TARGET_COLUMN].isna()].copy()
    dates = sorted(known["forecast_date"].dropna().unique())
    if len(dates) < 3:
        raise ValueError("Not enough forecast dates for a time-based train/test split.")

    cutoff_index = max(1, int(len(dates) * (1 - test_size)))
    cutoff_date = pd.Timestamp(dates[cutoff_index])
    train = known[known["forecast_date"] < cutoff_date].copy()
    test = known[known["forecast_date"] >= cutoff_date].copy()

    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty partition.")
    return train, test, future, cutoff_date


def fit_historical_same_day_average(train: pd.DataFrame) -> dict[str, Any]:
    """Fit same-day-of-week historical averages using training data only."""

    train_target = train[TARGET_COLUMN].astype(int)
    train = train.assign(_target=train_target)
    by_oblast_dow = (
        train.groupby(["oblast_name", "day_of_week"])["_target"].mean().to_dict()
    )
    by_dow = train.groupby("day_of_week")["_target"].mean().to_dict()
    global_mean = float(train["_target"].mean())
    return {
        "by_oblast_dow": by_oblast_dow,
        "by_dow": by_dow,
        "global_mean": global_mean,
    }


def predict_historical_same_day_average(
    dataframe: pd.DataFrame,
    model: dict[str, Any],
) -> pd.Series:
    """Predict with the fitted same-day-of-week average baseline."""

    probabilities: list[float] = []
    by_oblast_dow = model["by_oblast_dow"]
    by_dow = model["by_dow"]
    global_mean = model["global_mean"]

    for row in dataframe.itertuples(index=False):
        key = (row.oblast_name, row.day_of_week)
        probability = by_oblast_dow.get(
            key,
            by_dow.get(row.day_of_week, global_mean),
        )
        probabilities.append(float(probability))
    return pd.Series(probabilities, index=dataframe.index)


def build_classifier(model_type: str = MODEL_NAME) -> Pipeline:
    """Build a simple explainable sklearn classifier."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ]
    )

    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(
            "model_type must be 'logistic_regression' or 'random_forest'."
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_predictions(
    y_true: pd.Series,
    probabilities: pd.Series | np.ndarray,
    *,
    threshold: float = RISK_THRESHOLD,
) -> dict[str, float | None]:
    """Evaluate binary predictions and handle invalid ROC-AUC cases."""

    y_true_int = y_true.astype(int)
    probabilities_array = np.asarray(probabilities, dtype=float)
    y_pred = (probabilities_array >= threshold).astype(int)

    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true_int, y_pred)),
        "precision": float(precision_score(y_true_int, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_int, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_int, y_pred, zero_division=0)),
        "roc_auc": None,
    }

    if y_true_int.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true_int, probabilities_array))
    return metrics


def add_prediction_columns(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    future: pd.DataFrame,
    classifier: Pipeline,
    same_day_model: dict[str, Any],
) -> pd.DataFrame:
    """Add baseline and sklearn predictions to train/test/future rows."""

    predictions = pd.concat(
        [
            train.assign(split="train"),
            test.assign(split="test"),
            future.assign(split="future"),
        ],
        ignore_index=True,
    )

    rolling_probability = predictions["rolling_alert_probability_7d"].fillna(
        train[TARGET_COLUMN].astype(float).mean()
    )
    predictions["rolling_7d_probability"] = rolling_probability.clip(0, 1)
    predictions["rolling_7d_prediction"] = (
        predictions["rolling_7d_probability"] >= RISK_THRESHOLD
    ).astype(int)

    same_day_probability = predict_historical_same_day_average(
        predictions,
        same_day_model,
    )
    predictions["same_day_of_week_probability"] = same_day_probability.clip(0, 1)
    predictions["same_day_of_week_prediction"] = (
        predictions["same_day_of_week_probability"] >= RISK_THRESHOLD
    ).astype(int)

    predictions["model_probability"] = classifier.predict_proba(
        predictions[FEATURE_COLUMNS]
    )[:, 1]
    predictions["model_prediction"] = (
        predictions["model_probability"] >= RISK_THRESHOLD
    ).astype(int)

    output_columns = unique_preserving_order(
        [
        "observation_date",
        "forecast_date",
        "oblast_name",
        "split",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
        "rolling_7d_probability",
        "rolling_7d_prediction",
        "same_day_of_week_probability",
        "same_day_of_week_prediction",
        "model_probability",
        "model_prediction",
        ]
    )
    return predictions[output_columns].sort_values(["forecast_date", "oblast_name"])


def unique_preserving_order(values: list[str]) -> list[str]:
    """Return unique strings while preserving first-seen order."""

    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def run_forecast(
    *,
    daily_path: Path = DAILY_TIMESERIES_PATH,
    event_calendar_path: Path = EVENT_CALENDAR_PATH,
    output_path: Path = FORECAST_OUTPUT_PATH,
    metrics_path: Path = FORECAST_METRICS_PATH,
    model_type: str = MODEL_NAME,
    test_size: float = DEFAULT_TEST_SIZE,
) -> ForecastResult:
    """Run the full exploratory forecast pipeline."""

    daily = load_daily_timeseries(daily_path)
    event_calendar = load_event_calendar(event_calendar_path)
    features = build_features(daily, event_calendar)
    train, test, future, cutoff_date = split_train_test(features, test_size=test_size)

    y_train = train[TARGET_COLUMN].astype(int)
    y_test = test[TARGET_COLUMN].astype(int)

    same_day_model = fit_historical_same_day_average(train)

    classifier = build_classifier(model_type)
    classifier.fit(train[FEATURE_COLUMNS], y_train)

    rolling_test_prob = test["rolling_alert_probability_7d"].fillna(y_train.mean())
    same_day_test_prob = predict_historical_same_day_average(test, same_day_model)
    model_test_prob = classifier.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    metrics = {
        "framing": (
            "Exploratory baseline for next-day oblast-level alert-risk comparison; "
            "not for operational use."
        ),
        "target": TARGET_COLUMN,
        "model_type": model_type,
        "risk_threshold": RISK_THRESHOLD,
        "time_split": {
            "test_size": test_size,
            "cutoff_forecast_date": cutoff_date.date().isoformat(),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "future_rows": int(len(future)),
            "train_date_min": train["forecast_date"].min().date().isoformat(),
            "train_date_max": train["forecast_date"].max().date().isoformat(),
            "test_date_min": test["forecast_date"].min().date().isoformat(),
            "test_date_max": test["forecast_date"].max().date().isoformat(),
        },
        "class_balance": {
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
        "features": FEATURE_COLUMNS,
        "metrics": {
            "rolling_7d_probability": evaluate_predictions(y_test, rolling_test_prob),
            "historical_same_day_of_week": evaluate_predictions(
                y_test,
                same_day_test_prob,
            ),
            model_type: evaluate_predictions(y_test, model_test_prob),
        },
    }

    predictions = add_prediction_columns(
        train=train,
        test=test,
        future=future,
        classifier=classifier,
        same_day_model=same_day_model,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return ForecastResult(
        predictions=predictions,
        metrics=metrics,
        output_path=output_path,
        metrics_path=metrics_path,
    )


def print_forecast_summary(result: ForecastResult) -> None:
    """Print metrics and generated output paths."""

    print("Exploratory baseline forecast complete")
    print("Framing: not for operational use.")
    print(f"Predictions saved: {result.output_path}")
    print(f"Metrics saved: {result.metrics_path}")
    print("\nMetrics")
    for name, metrics in result.metrics["metrics"].items():
        metric_parts = []
        for metric_name, value in metrics.items():
            rendered = "n/a" if value is None else f"{value:.4f}"
            metric_parts.append(f"{metric_name}={rendered}")
        print(f"- {name}: " + ", ".join(metric_parts))

    future = result.predictions[result.predictions["split"] == "future"]
    if not future.empty:
        print("\nNext-day forecast rows")
        print(
            future[
                [
                    "forecast_date",
                    "oblast_name",
                    "model_probability",
                    "rolling_7d_probability",
                    "same_day_of_week_probability",
                    "is_event_window",
                    "event_category",
                ]
            ]
            .sort_values("model_probability", ascending=False)
            .head(10)
            .to_string(index=False)
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exploratory next-day alert-risk baselines."
    )
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "random_forest"],
        default=MODEL_NAME,
        help="Simple sklearn model to train. Default: logistic_regression.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of latest dates used for test split. Default: 0.2.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_forecast(model_type=args.model, test_size=args.test_size)
    print_forecast_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
