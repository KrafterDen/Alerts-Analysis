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
FORECAST_60D_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_60d_oblast.csv"
FORECAST_60D_METRICS_PATH = PROJECT_ROOT / "outputs" / "tables" / "forecast_60d_metrics.json"
MODEL_ARTIFACTS_DIR = PROJECT_ROOT / "outputs" / "models"

TARGET_COLUMN = "had_alert_binary_next_day"
MODEL_NAME = "logistic_regression"
DEFAULT_TEST_SIZE = 0.2
RISK_THRESHOLD = 0.5
DEFAULT_FORECAST_HORIZON = 60
DEFAULT_LOOKBACK_DAYS = 90
EVENT_UPLIFT_MIN = 0.75
EVENT_UPLIFT_MAX = 1.50
MILESTONE_DAYS = [1, 7, 14, 30, 60]

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


@dataclass(frozen=True)
class Forecast60DResult:
    """60-day forecast output payload for CLI and downstream reuse."""

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


def build_event_detail_lookup(event_calendar: pd.DataFrame | None) -> dict[pd.Timestamp, dict[str, str | int]]:
    """Map forecast dates to analytical event-window labels."""

    if event_calendar is None or event_calendar.empty:
        return {}

    lookup: dict[pd.Timestamp, dict[str, set[str]]] = {}
    for row in event_calendar.itertuples(index=False):
        start = pd.Timestamp(row.window_start_date).normalize()
        end = pd.Timestamp(row.window_end_date).normalize()
        event_name = str(getattr(row, "name", ""))
        category = str(getattr(row, "category", ""))
        for day in pd.date_range(start, end, freq="D"):
            entry = lookup.setdefault(day, {"categories": set(), "names": set()})
            if category:
                entry["categories"].add(category)
            if event_name:
                entry["names"].add(event_name)

    details: dict[pd.Timestamp, dict[str, str | int]] = {}
    for day, values in lookup.items():
        categories = sorted(values["categories"])
        names = sorted(values["names"])
        details[day] = {
            "is_event_window": 1,
            "event_category": collapse_event_categories(categories),
            "event_name": "; ".join(names),
        }
    return details


def event_detail_for_date(
    forecast_date: pd.Timestamp,
    event_lookup: dict[pd.Timestamp, dict[str, str | int]],
) -> dict[str, str | int]:
    detail = event_lookup.get(pd.Timestamp(forecast_date).normalize())
    if detail is None:
        return {
            "is_event_window": 0,
            "event_category": "none",
            "event_name": "",
        }
    return detail


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


def prepare_completed_history(
    daily: pd.DataFrame,
    event_calendar: pd.DataFrame | None,
) -> pd.DataFrame:
    """Create a complete historical grid with zero-alert days and event labels."""

    history = complete_daily_grid(daily, include_future_day=False)
    history = history.rename(columns={"forecast_date": "date"})
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["day_of_week"] = history["date"].dt.dayofweek
    history["month"] = history["date"].dt.month
    event_lookup = build_event_detail_lookup(event_calendar)
    details = history["date"].map(lambda value: event_detail_for_date(value, event_lookup))
    history["is_event_window"] = details.map(lambda value: int(value["is_event_window"]))
    history["event_category"] = details.map(lambda value: str(value["event_category"]))
    history["event_name"] = details.map(lambda value: str(value["event_name"]))
    for column in ["alert_count", "total_alert_minutes", "had_alert_binary"]:
        history[column] = pd.to_numeric(history[column], errors="coerce").fillna(0)
    return history


def future_forecast_frame(
    history: pd.DataFrame,
    event_calendar: pd.DataFrame | None,
    *,
    horizon: int,
) -> pd.DataFrame:
    """Build the future oblast-date grid used by all 60-day models."""

    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    origin_date = pd.Timestamp(history["date"].max()).normalize()
    forecast_dates = pd.date_range(
        origin_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )
    oblasts = sorted(history["oblast_name"].dropna().unique())
    future = pd.MultiIndex.from_product(
        [oblasts, forecast_dates],
        names=["oblast_name", "forecast_date"],
    ).to_frame(index=False)
    future["forecast_origin_date"] = origin_date
    future["horizon_day"] = (
        future["forecast_date"] - future["forecast_origin_date"]
    ).dt.days.astype(int)
    future["day_of_week"] = future["forecast_date"].dt.dayofweek
    future["month"] = future["forecast_date"].dt.month

    event_lookup = build_event_detail_lookup(event_calendar)
    details = future["forecast_date"].map(
        lambda value: event_detail_for_date(value, event_lookup)
    )
    future["is_event_window"] = details.map(lambda value: int(value["is_event_window"]))
    future["event_category"] = details.map(lambda value: str(value["event_category"]))
    future["event_name"] = details.map(lambda value: str(value["event_name"]))
    return future


def fit_multitarget_same_day_averages(history: pd.DataFrame) -> dict[str, Any]:
    """Fit same-day-of-week averages for risk, count, and minutes."""

    targets = {
        "predicted_alert_probability": "had_alert_binary",
        "predicted_alert_count": "alert_count",
        "predicted_alert_minutes": "total_alert_minutes",
    }
    model: dict[str, Any] = {"targets": {}}
    for output_column, source_column in targets.items():
        model["targets"][output_column] = {
            "by_oblast_dow": history.groupby(["oblast_name", "day_of_week"])[
                source_column
            ].mean().to_dict(),
            "by_dow": history.groupby("day_of_week")[source_column].mean().to_dict(),
            "by_oblast": history.groupby("oblast_name")[source_column].mean().to_dict(),
            "global": float(history[source_column].mean()),
        }
    return model


def predict_same_day_values(future: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    """Predict risk, count, and minutes from fitted same-weekday averages."""

    output = pd.DataFrame(index=future.index)
    for output_column, target_model in model["targets"].items():
        values: list[float] = []
        for row in future.itertuples(index=False):
            key = (row.oblast_name, row.day_of_week)
            value = target_model["by_oblast_dow"].get(
                key,
                target_model["by_dow"].get(
                    row.day_of_week,
                    target_model["by_oblast"].get(row.oblast_name, target_model["global"]),
                ),
            )
            values.append(float(value))
        output[output_column] = values
    output["predicted_alert_probability"] = output[
        "predicted_alert_probability"
    ].clip(0, 1)
    output["predicted_alert_count"] = output["predicted_alert_count"].clip(lower=0)
    output["predicted_alert_minutes"] = output["predicted_alert_minutes"].clip(lower=0)
    return output


def build_rolling_7d_baseline(history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """Forecast each future day from the latest trailing 7-day oblast average."""

    recent = (
        history.sort_values("date")
        .groupby("oblast_name", group_keys=False)
        .tail(7)
        .groupby("oblast_name", as_index=False)
        .agg(
            predicted_alert_probability=("had_alert_binary", "mean"),
            predicted_alert_count=("alert_count", "mean"),
            predicted_alert_minutes=("total_alert_minutes", "mean"),
        )
    )
    output = future.merge(recent, on="oblast_name", how="left")
    for column in [
        "predicted_alert_probability",
        "predicted_alert_count",
        "predicted_alert_minutes",
    ]:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0)
    output["predicted_alert_probability"] = output[
        "predicted_alert_probability"
    ].clip(0, 1)
    output["model_name"] = "baseline_rolling_7d"
    return select_forecast_60d_columns(output)


def calculate_event_uplift_model(history: pd.DataFrame) -> dict[str, Any]:
    """Estimate capped event-window uplift factors from historical data."""

    source_to_output = {
        "had_alert_binary": "predicted_alert_probability",
        "alert_count": "predicted_alert_count",
        "total_alert_minutes": "predicted_alert_minutes",
    }
    non_event = history[history["is_event_window"].eq(0)]
    event_rows = history[
        history["is_event_window"].eq(1) & history["event_category"].ne("none")
    ]
    model: dict[str, Any] = {}
    for source_column, output_column in source_to_output.items():
        global_base = float(non_event[source_column].mean()) if not non_event.empty else 0.0
        global_event = float(event_rows[source_column].mean()) if not event_rows.empty else global_base
        global_multiplier = capped_ratio(global_event, global_base)
        model[output_column] = {
            "by_oblast_category": {},
            "by_category": {},
            "global": global_multiplier,
        }

        category_means = event_rows.groupby("event_category")[source_column].mean()
        for category, event_mean in category_means.items():
            model[output_column]["by_category"][category] = capped_ratio(
                float(event_mean),
                global_base,
            )

        non_event_oblast = non_event.groupby("oblast_name")[source_column].mean()
        event_oblast_category = event_rows.groupby(["oblast_name", "event_category"])[
            source_column
        ].mean()
        for key, event_mean in event_oblast_category.items():
            oblast, category = key
            base = float(non_event_oblast.get(oblast, global_base))
            model[output_column]["by_oblast_category"][key] = capped_ratio(
                float(event_mean),
                base,
            )
    return model


def capped_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 1.0
    return float(np.clip(numerator / denominator, EVENT_UPLIFT_MIN, EVENT_UPLIFT_MAX))


def event_multiplier(
    model: dict[str, Any],
    output_column: str,
    oblast: str,
    category: str,
) -> float:
    if not category or category == "none":
        return 1.0
    target_model = model.get(output_column, {})
    return float(
        target_model.get("by_oblast_category", {}).get(
            (oblast, category),
            target_model.get("by_category", {}).get(
                category,
                target_model.get("global", 1.0),
            ),
        )
    )


def build_same_day_baseline(history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    same_day_model = fit_multitarget_same_day_averages(history)
    predictions = pd.concat(
        [future.reset_index(drop=True), predict_same_day_values(future, same_day_model)],
        axis=1,
    )
    predictions["model_name"] = "baseline_same_day_of_week"
    return select_forecast_60d_columns(predictions)


def build_event_adjusted_baseline(history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    same_day_model = fit_multitarget_same_day_averages(history)
    uplift_model = calculate_event_uplift_model(history)
    predictions = pd.concat(
        [future.reset_index(drop=True), predict_same_day_values(future, same_day_model)],
        axis=1,
    )
    for index, row in predictions.iterrows():
        if int(row["is_event_window"]) != 1:
            continue
        for output_column in [
            "predicted_alert_probability",
            "predicted_alert_count",
            "predicted_alert_minutes",
        ]:
            multiplier = event_multiplier(
                uplift_model,
                output_column,
                str(row["oblast_name"]),
                str(row["event_category"]),
            )
            predictions.at[index, output_column] = float(row[output_column]) * multiplier

    predictions["predicted_alert_probability"] = predictions[
        "predicted_alert_probability"
    ].clip(0, 1)
    predictions["predicted_alert_count"] = predictions["predicted_alert_count"].clip(lower=0)
    predictions["predicted_alert_minutes"] = predictions[
        "predicted_alert_minutes"
    ].clip(lower=0)
    predictions["model_name"] = "baseline_event_adjusted"
    return select_forecast_60d_columns(predictions)


def select_forecast_60d_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "forecast_date",
        "oblast_name",
        "model_name",
        "predicted_alert_probability",
        "predicted_alert_count",
        "predicted_alert_minutes",
        "is_event_window",
        "event_category",
        "event_name",
        "forecast_origin_date",
        "horizon_day",
    ]
    output = dataframe[columns].copy()
    output["forecast_date"] = pd.to_datetime(output["forecast_date"], errors="coerce")
    output["forecast_origin_date"] = pd.to_datetime(
        output["forecast_origin_date"], errors="coerce"
    )
    output["predicted_alert_probability"] = pd.to_numeric(
        output["predicted_alert_probability"], errors="coerce"
    ).clip(0, 1)
    output["predicted_alert_count"] = pd.to_numeric(
        output["predicted_alert_count"], errors="coerce"
    ).clip(lower=0)
    output["predicted_alert_minutes"] = pd.to_numeric(
        output["predicted_alert_minutes"], errors="coerce"
    ).clip(lower=0)
    output["is_event_window"] = pd.to_numeric(
        output["is_event_window"], errors="coerce"
    ).fillna(0).astype(int)
    output["event_category"] = output["event_category"].fillna("none")
    output["event_name"] = output["event_name"].fillna("")
    output["horizon_day"] = pd.to_numeric(
        output["horizon_day"], errors="coerce"
    ).astype(int)
    return output


def build_baseline_forecast_60d(
    history: pd.DataFrame,
    future: pd.DataFrame,
) -> pd.DataFrame:
    outputs = [
        build_rolling_7d_baseline(history, future),
        build_same_day_baseline(history, future),
        build_event_adjusted_baseline(history, future),
    ]
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["model_name", "forecast_date", "oblast_name"]
    )


def build_lstm_feature_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Prepare scaled sequence features for the optional LSTM."""

    features = history.sort_values(["oblast_name", "date"]).copy()
    features["log_alert_count"] = np.log1p(features["alert_count"].clip(lower=0))
    features["log_alert_minutes"] = np.log1p(
        features["total_alert_minutes"].clip(lower=0)
    )
    features["dow_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
    features["dow_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
    return features


def build_lstm_training_arrays(
    history: pd.DataFrame,
    *,
    horizon: int,
    lookback_days: int,
) -> dict[str, Any]:
    feature_frame = build_lstm_feature_frame(history)
    sequence_columns = [
        "log_alert_count",
        "log_alert_minutes",
        "had_alert_binary",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_event_window",
    ]
    oblasts = sorted(feature_frame["oblast_name"].dropna().unique())
    oblast_to_id = {oblast: index for index, oblast in enumerate(oblasts)}

    x_values: list[np.ndarray] = []
    oblast_ids: list[int] = []
    y_probability: list[np.ndarray] = []
    y_count: list[np.ndarray] = []
    y_minutes: list[np.ndarray] = []
    origin_dates: list[pd.Timestamp] = []

    for oblast, rows in feature_frame.groupby("oblast_name", sort=True):
        rows = rows.sort_values("date").reset_index(drop=True)
        if len(rows) < lookback_days + horizon:
            continue
        sequence_data = rows[sequence_columns].to_numpy(dtype=np.float32)
        probability_target = rows["had_alert_binary"].to_numpy(dtype=np.float32)
        count_target = np.log1p(
            rows["alert_count"].clip(lower=0).to_numpy(dtype=np.float32)
        )
        minutes_target = np.log1p(
            rows["total_alert_minutes"].clip(lower=0).to_numpy(dtype=np.float32)
        )
        for origin_index in range(lookback_days - 1, len(rows) - horizon):
            start_index = origin_index - lookback_days + 1
            target_start = origin_index + 1
            target_end = target_start + horizon
            x_values.append(sequence_data[start_index : origin_index + 1])
            oblast_ids.append(oblast_to_id[oblast])
            y_probability.append(probability_target[target_start:target_end])
            y_count.append(count_target[target_start:target_end])
            y_minutes.append(minutes_target[target_start:target_end])
            origin_dates.append(pd.Timestamp(rows.loc[origin_index, "date"]).normalize())

    if not x_values:
        raise ValueError("Not enough history to train the LSTM forecast model.")

    return {
        "x": np.stack(x_values).astype(np.float32),
        "oblast_ids": np.asarray(oblast_ids, dtype=np.int64),
        "y_probability": np.stack(y_probability).astype(np.float32),
        "y_count": np.stack(y_count).astype(np.float32),
        "y_minutes": np.stack(y_minutes).astype(np.float32),
        "origin_dates": np.asarray(origin_dates, dtype="datetime64[ns]"),
        "oblasts": oblasts,
        "sequence_columns": sequence_columns,
    }


def try_build_lstm_forecast_60d(
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    horizon: int,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    max_epochs: int = 100,
    patience: int = 10,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Train an optional PyTorch LSTM and return daily 60-day predictions."""

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        return None, {
            "status": "skipped",
            "reason": f"PyTorch is not installed: {exc}",
        }

    torch.manual_seed(42)
    np.random.seed(42)

    arrays = build_lstm_training_arrays(
        history,
        horizon=horizon,
        lookback_days=lookback_days,
    )

    class MultiTaskLSTM(nn.Module):
        def __init__(
            self,
            *,
            input_size: int,
            oblast_count: int,
            horizon_days: int,
            hidden_size: int = 64,
            embedding_size: int = 8,
        ) -> None:
            super().__init__()
            self.embedding = nn.Embedding(oblast_count, embedding_size)
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=2,
                dropout=0.2,
                batch_first=True,
            )
            self.shared = nn.Sequential(
                nn.Linear(hidden_size + embedding_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.probability_head = nn.Linear(hidden_size, horizon_days)
            self.count_head = nn.Linear(hidden_size, horizon_days)
            self.minutes_head = nn.Linear(hidden_size, horizon_days)

        def forward(self, sequences: Any, oblast_ids: Any) -> tuple[Any, Any, Any]:
            _, (hidden, _) = self.lstm(sequences)
            last_hidden = hidden[-1]
            oblast_embedding = self.embedding(oblast_ids)
            shared = self.shared(torch.cat([last_hidden, oblast_embedding], dim=1))
            return (
                self.probability_head(shared),
                self.count_head(shared),
                self.minutes_head(shared),
            )

    origin_dates = pd.to_datetime(arrays["origin_dates"])
    unique_origins = sorted(origin_dates.unique())
    cutoff_index = max(1, int(len(unique_origins) * 0.8))
    validation_start = unique_origins[cutoff_index]
    train_mask = origin_dates < validation_start
    validation_mask = ~train_mask
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("LSTM time split produced an empty train or validation set.")

    def make_dataset(mask: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.tensor(arrays["x"][mask], dtype=torch.float32),
            torch.tensor(arrays["oblast_ids"][mask], dtype=torch.long),
            torch.tensor(arrays["y_probability"][mask], dtype=torch.float32),
            torch.tensor(arrays["y_count"][mask], dtype=torch.float32),
            torch.tensor(arrays["y_minutes"][mask], dtype=torch.float32),
        )

    train_loader = DataLoader(
        make_dataset(train_mask),
        batch_size=64,
        shuffle=True,
    )
    validation_loader = DataLoader(
        make_dataset(validation_mask),
        batch_size=128,
        shuffle=False,
    )

    model = MultiTaskLSTM(
        input_size=arrays["x"].shape[-1],
        oblast_count=len(arrays["oblasts"]),
        horizon_days=horizon,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    probability_loss = nn.BCEWithLogitsLoss()
    regression_loss = nn.MSELoss()

    best_state: dict[str, Any] | None = None
    best_validation_loss = float("inf")
    stale_epochs = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for sequences, oblast_ids, y_probability, y_count, y_minutes in train_loader:
            optimizer.zero_grad()
            probability_logits, count_log, minutes_log = model(sequences, oblast_ids)
            loss = (
                probability_loss(probability_logits, y_probability)
                + regression_loss(count_log, y_count)
                + regression_loss(minutes_log, y_minutes)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for sequences, oblast_ids, y_probability, y_count, y_minutes in validation_loader:
                probability_logits, count_log, minutes_log = model(sequences, oblast_ids)
                loss = (
                    probability_loss(probability_logits, y_probability)
                    + regression_loss(count_log, y_count)
                    + regression_loss(minutes_log, y_minutes)
                )
                validation_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    future_sequences, future_oblast_ids = build_lstm_future_inputs(
        history,
        arrays["oblasts"],
        lookback_days=lookback_days,
        sequence_columns=arrays["sequence_columns"],
    )
    model.eval()
    with torch.no_grad():
        probability_logits, count_log, minutes_log = model(
            torch.tensor(future_sequences, dtype=torch.float32),
            torch.tensor(future_oblast_ids, dtype=torch.long),
        )
    probability = torch.sigmoid(probability_logits).cpu().numpy()
    count = np.expm1(count_log.cpu().numpy()).clip(min=0)
    minutes = np.expm1(minutes_log.cpu().numpy()).clip(min=0)

    predictions = lstm_predictions_to_frame(
        future=future,
        oblasts=arrays["oblasts"],
        probability=probability,
        count=count,
        minutes=minutes,
    )

    MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_ARTIFACTS_DIR / "lstm_multitask_v1.pt"
    metadata_path = MODEL_ARTIFACTS_DIR / "lstm_multitask_v1_metadata.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "oblasts": arrays["oblasts"],
            "sequence_columns": arrays["sequence_columns"],
            "horizon": horizon,
            "lookback_days": lookback_days,
        },
        model_path,
    )
    metadata = {
        "status": "trained",
        "model_name": "lstm_multitask_v1",
        "lookback_days": lookback_days,
        "horizon": horizon,
        "hidden_size": 64,
        "layers": 2,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "max_epochs": max_epochs,
        "epochs_trained": len(history_rows),
        "best_validation_loss": best_validation_loss,
        "validation_start_origin": pd.Timestamp(validation_start).date().isoformat(),
        "model_path": str(model_path),
        "training_history": history_rows,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return predictions, metadata


def build_lstm_future_inputs(
    history: pd.DataFrame,
    oblasts: list[str],
    *,
    lookback_days: int,
    sequence_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    feature_frame = build_lstm_feature_frame(history)
    sequences: list[np.ndarray] = []
    oblast_ids: list[int] = []
    for oblast_id, oblast in enumerate(oblasts):
        rows = feature_frame[feature_frame["oblast_name"].eq(oblast)].sort_values("date")
        if len(rows) < lookback_days:
            raise ValueError(f"Not enough history for LSTM forecast: {oblast}")
        sequences.append(rows.tail(lookback_days)[sequence_columns].to_numpy(dtype=np.float32))
        oblast_ids.append(oblast_id)
    return np.stack(sequences).astype(np.float32), np.asarray(oblast_ids, dtype=np.int64)


def lstm_predictions_to_frame(
    *,
    future: pd.DataFrame,
    oblasts: list[str],
    probability: np.ndarray,
    count: np.ndarray,
    minutes: np.ndarray,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for oblast_index, oblast in enumerate(oblasts):
        oblast_future = future[future["oblast_name"].eq(oblast)].sort_values(
            "forecast_date"
        ).copy()
        oblast_future["predicted_alert_probability"] = probability[oblast_index][
            : len(oblast_future)
        ]
        oblast_future["predicted_alert_count"] = count[oblast_index][
            : len(oblast_future)
        ]
        oblast_future["predicted_alert_minutes"] = minutes[oblast_index][
            : len(oblast_future)
        ]
        oblast_future["model_name"] = "lstm_multitask_v1"
        rows.append(oblast_future)
    return select_forecast_60d_columns(pd.concat(rows, ignore_index=True))


def parse_requested_models(models: str | list[str]) -> list[str]:
    if isinstance(models, list):
        requested = [str(value).strip() for value in models]
    else:
        requested = [value.strip() for value in str(models).split(",")]
    normalized = [value for value in requested if value]
    allowed = {"baselines", "lstm"}
    invalid = sorted(set(normalized) - allowed)
    if invalid:
        raise ValueError(f"Unsupported 60-day forecast models: {invalid}")
    if not normalized:
        return ["baselines"]
    return unique_preserving_order(normalized)


def choose_backtest_origins(
    history: pd.DataFrame,
    *,
    horizon: int,
    lookback_days: int,
    max_origins: int = 4,
) -> list[pd.Timestamp]:
    dates = sorted(pd.to_datetime(history["date"]).dropna().unique())
    if not dates:
        return []
    earliest_index = lookback_days - 1
    latest_index = len(dates) - horizon - 1
    if latest_index < earliest_index:
        return []
    candidate_indices = list(range(earliest_index, latest_index + 1, 30))
    if not candidate_indices or candidate_indices[-1] != latest_index:
        candidate_indices.append(latest_index)
    chosen = candidate_indices[-max_origins:]
    return [pd.Timestamp(dates[index]).normalize() for index in chosen]


def forecast_from_truncated_history(
    history: pd.DataFrame,
    event_calendar: pd.DataFrame | None,
    *,
    origin_date: pd.Timestamp,
    horizon: int,
    requested_models: list[str],
) -> pd.DataFrame:
    truncated = history[history["date"].le(origin_date)].copy()
    if truncated.empty:
        return pd.DataFrame()
    future = future_forecast_frame(truncated, event_calendar, horizon=horizon)
    
    frames: list[pd.DataFrame] = []
    if "baselines" in requested_models:
        frames.append(build_baseline_forecast_60d(truncated, future))
        
    if "lstm" in requested_models:
        lstm_predictions, _ = try_build_lstm_forecast_60d(
            truncated,
            future,
            horizon=horizon,
            lookback_days=DEFAULT_LOOKBACK_DAYS,
        )
        if lstm_predictions is not None and not lstm_predictions.empty:
            frames.append(lstm_predictions)
            
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def evaluate_forecast_60d_predictions(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    milestone_days: list[int] | None = None,
) -> dict[str, Any]:
    milestones = milestone_days or MILESTONE_DAYS
    actual_frame = actuals[
        ["date", "oblast_name", "had_alert_binary", "alert_count", "total_alert_minutes"]
    ].rename(columns={"date": "forecast_date"})
    merged = predictions.merge(
        actual_frame,
        on=["forecast_date", "oblast_name"],
        how="left",
    )

    metrics: dict[str, Any] = {}
    for model_name, model_rows in merged.groupby("model_name", dropna=False):
        per_horizon: dict[str, Any] = {}
        for day in milestones:
            horizon_rows = model_rows[model_rows["horizon_day"].eq(day)].copy()
            if horizon_rows.empty:
                continue
            actual_probability = pd.to_numeric(
                horizon_rows["had_alert_binary"], errors="coerce"
            )
            predicted_probability = pd.to_numeric(
                horizon_rows["predicted_alert_probability"], errors="coerce"
            ).clip(0, 1)
            probability_mae = float(
                np.abs(predicted_probability - actual_probability).mean()
            )
            probability_brier = float(
                np.square(predicted_probability - actual_probability).mean()
            )
            count_mae = float(
                np.abs(
                    pd.to_numeric(horizon_rows["predicted_alert_count"], errors="coerce")
                    - pd.to_numeric(horizon_rows["alert_count"], errors="coerce")
                ).mean()
            )
            minutes_mae = float(
                np.abs(
                    pd.to_numeric(horizon_rows["predicted_alert_minutes"], errors="coerce")
                    - pd.to_numeric(horizon_rows["total_alert_minutes"], errors="coerce")
                ).mean()
            )
            per_horizon[str(day)] = {
                "rows": int(len(horizon_rows)),
                "probability_mae": probability_mae,
                "probability_brier": probability_brier,
                "count_mae": count_mae,
                "minutes_mae": minutes_mae,
            }
        metrics[str(model_name)] = per_horizon
    return metrics


def run_backtests(
    history: pd.DataFrame,
    event_calendar: pd.DataFrame | None,
    *,
    horizon: int,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    requested_models: list[str],
) -> dict[str, Any]:
    origins = choose_backtest_origins(
        history,
        horizon=horizon,
        lookback_days=lookback_days,
    )
    if not origins:
        return {
            "origins": [],
            "metrics_by_model": {},
        }

    prediction_frames: list[pd.DataFrame] = []
    actual_frames: list[pd.DataFrame] = []
    for origin in origins:
        forecast_frame = forecast_from_truncated_history(
            history,
            event_calendar,
            origin_date=origin,
            horizon=horizon,
            requested_models=requested_models,
        )
        if forecast_frame.empty:
            continue
        actuals = history[
            history["date"].between(
                origin + pd.Timedelta(days=1),
                origin + pd.Timedelta(days=horizon),
            )
        ].copy()
        prediction_frames.append(forecast_frame)
        actual_frames.append(actuals)

    if not prediction_frames or not actual_frames:
        return {
            "origins": [origin.date().isoformat() for origin in origins],
            "metrics_by_model": {},
        }

    predictions = pd.concat(prediction_frames, ignore_index=True)
    actuals = pd.concat(actual_frames, ignore_index=True)
    return {
        "origins": [origin.date().isoformat() for origin in origins],
        "metrics_by_model": evaluate_forecast_60d_predictions(predictions, actuals),
    }


def run_forecast_60d(
    *,
    daily_path: Path = DAILY_TIMESERIES_PATH,
    event_calendar_path: Path = EVENT_CALENDAR_PATH,
    output_path: Path = FORECAST_60D_OUTPUT_PATH,
    metrics_path: Path = FORECAST_60D_METRICS_PATH,
    horizon: int = DEFAULT_FORECAST_HORIZON,
    models: str | list[str] = "baselines",
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> Forecast60DResult:
    daily = load_daily_timeseries(daily_path)
    event_calendar = load_event_calendar(event_calendar_path)
    history = prepare_completed_history(daily, event_calendar)
    future = future_forecast_frame(history, event_calendar, horizon=horizon)
    requested_models = parse_requested_models(models)

    output_frames: list[pd.DataFrame] = []
    lstm_metadata: dict[str, Any] = {
        "status": "not_requested",
    }

    if "baselines" in requested_models:
        output_frames.append(build_baseline_forecast_60d(history, future))

    if "lstm" in requested_models:
        lstm_predictions, lstm_metadata = try_build_lstm_forecast_60d(
            history,
            future,
            horizon=horizon,
            lookback_days=lookback_days,
        )
        if lstm_predictions is not None and not lstm_predictions.empty:
            output_frames.append(lstm_predictions)

    if not output_frames:
        raise ValueError("No 60-day forecast outputs were generated.")

    predictions = pd.concat(output_frames, ignore_index=True).sort_values(
        ["model_name", "forecast_date", "oblast_name"]
    )

    validation = run_backtests(
        history,
        event_calendar,
        horizon=horizon,
        lookback_days=lookback_days,
        requested_models=requested_models,
    )
    metrics = {
        "framing": (
            "Exploratory 60-day oblast-level forecast comparison only; "
            "not for operational use."
        ),
        "forecast_origin_date": pd.Timestamp(history["date"].max()).date().isoformat(),
        "horizon_days": int(horizon),
        "lookback_days": int(lookback_days),
        "requested_models": requested_models,
        "generated_models": sorted(predictions["model_name"].dropna().unique()),
        "rows": int(len(predictions)),
        "future_date_min": predictions["forecast_date"].min().date().isoformat(),
        "future_date_max": predictions["forecast_date"].max().date().isoformat(),
        "milestone_days": MILESTONE_DAYS,
        "backtests": validation,
        "lstm": lstm_metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return Forecast60DResult(
        predictions=predictions,
        metrics=metrics,
        output_path=output_path,
        metrics_path=metrics_path,
    )


def print_forecast_60d_summary(result: Forecast60DResult) -> None:
    print("Exploratory 60-day forecast complete")
    print("Framing: not for operational use.")
    print(f"Predictions saved: {result.output_path}")
    print(f"Metrics saved: {result.metrics_path}")
    print(f"Rows: {len(result.predictions):,}")
    print("Models: " + ", ".join(result.metrics["generated_models"]))
    print("\nMilestone backtests")
    metrics_by_model = result.metrics.get("backtests", {}).get("metrics_by_model", {})
    if not metrics_by_model:
        print("- No rolling backtests available.")
    for model_name, horizon_metrics in metrics_by_model.items():
        parts = []
        for horizon_day in MILESTONE_DAYS:
            metrics = horizon_metrics.get(str(horizon_day))
            if not metrics:
                continue
            parts.append(
                f"d{horizon_day}: prob_mae={metrics['probability_mae']:.4f}, "
                f"count_mae={metrics['count_mae']:.2f}, minutes_mae={metrics['minutes_mae']:.2f}"
            )
        if parts:
            print(f"- {model_name}: " + "; ".join(parts))

    future_preview = result.predictions[
        result.predictions["horizon_day"].eq(1)
    ].copy()
    if not future_preview.empty:
        print("\nDay-1 forecast preview")
        print(
            future_preview[
                [
                    "forecast_date",
                    "oblast_name",
                    "model_name",
                    "predicted_alert_probability",
                    "predicted_alert_count",
                    "predicted_alert_minutes",
                    "event_category",
                ]
            ]
            .sort_values(["model_name", "predicted_alert_probability"], ascending=[True, False])
            .head(12)
            .to_string(index=False)
        )


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
        description="Run exploratory alert-risk and 60-day forecast baselines."
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
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon in days. Use 1 for the existing next-day run, or 60 for the expanded future forecast.",
    )
    parser.add_argument(
        "--models",
        default="baselines",
        help="Comma-separated 60-day models to run. Supported: baselines,lstm.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.horizon <= 1:
        result = run_forecast(model_type=args.model, test_size=args.test_size)
        print_forecast_summary(result)
    else:
        result_60d = run_forecast_60d(
            horizon=args.horizon,
            models=args.models,
        )
        print_forecast_60d_summary(result_60d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
