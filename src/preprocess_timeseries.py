"""Build oblast-level time series from merged event-level alert records."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from merge_sources import MERGED_OUTPUT_PATH
except ImportError:
    from .merge_sources import MERGED_OUTPUT_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAILY_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "daily_oblast_timeseries.csv"
HOURLY_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_oblast_timeseries.csv"
PREPROCESS_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "timeseries_preprocess_report.json"

KYIV_TIMEZONE = "Europe/Kyiv"
REASONABLE_MAX_DURATION_MINUTES = 7 * 24 * 60
DATA_SOURCE_COLUMNS = ("github_official", "alerts_api")

REQUIRED_MERGED_COLUMNS = {
    "source_record_id",
    "data_source",
    "oblast_name",
    "location_type",
    "alert_type",
    "started_at_utc",
    "finished_at_utc",
    "duration_minutes",
    "is_finished",
}


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class ValidationResult:
    """Validation counts for preprocessing report."""

    input_rows: int
    duplicate_event_records: int
    missing_started_at: int
    missing_finished_at: int
    negative_duration_rows: int
    zero_duration_rows: int
    long_duration_rows: int
    impossible_date_rows: int
    max_duration_minutes: float | None


def load_merged_alerts(path: Path = MERGED_OUTPUT_PATH) -> pd.DataFrame:
    """Load merged event-level alerts and parse timestamps as UTC."""

    if not path.exists():
        raise FileNotFoundError(
            f"Merged event-level dataset not found: {path}. "
            "Run src/merge_sources.py first."
        )

    dataframe = pd.read_csv(path, dtype="string")
    require_columns(dataframe.columns)
    dataframe["started_at_utc"] = to_utc(dataframe["started_at_utc"])
    dataframe["finished_at_utc"] = to_utc(dataframe["finished_at_utc"])
    dataframe["duration_minutes"] = pd.to_numeric(
        dataframe["duration_minutes"],
        errors="coerce",
    )
    dataframe["is_finished"] = dataframe["finished_at_utc"].notna()
    return dataframe


def validate_merged_alerts(
    dataframe: pd.DataFrame,
    *,
    max_reasonable_duration_minutes: int = REASONABLE_MAX_DURATION_MINUTES,
) -> ValidationResult:
    """Run data quality checks before aggregation."""

    duplicate_event_records = int(dataframe.duplicated("source_record_id").sum())
    missing_started_at = int(dataframe["started_at_utc"].isna().sum())
    missing_finished_at = int(dataframe["finished_at_utc"].isna().sum())
    negative_duration_rows = int((dataframe["duration_minutes"] < 0).sum())
    zero_duration_rows = int((dataframe["duration_minutes"] == 0).sum())
    long_duration_rows = int(
        (dataframe["duration_minutes"] > max_reasonable_duration_minutes).sum()
    )

    impossible_date_rows = int(
        (
            dataframe["started_at_utc"].notna()
            & (
                (dataframe["started_at_utc"] < pd.Timestamp("2022-02-24", tz="UTC"))
                | (
                    dataframe["started_at_utc"]
                    > pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
                )
            )
        ).sum()
    )

    max_duration = dataframe["duration_minutes"].max(skipna=True)
    max_duration_value = None if pd.isna(max_duration) else float(max_duration)

    return ValidationResult(
        input_rows=len(dataframe),
        duplicate_event_records=duplicate_event_records,
        missing_started_at=missing_started_at,
        missing_finished_at=missing_finished_at,
        negative_duration_rows=negative_duration_rows,
        zero_duration_rows=zero_duration_rows,
        long_duration_rows=long_duration_rows,
        impossible_date_rows=impossible_date_rows,
        max_duration_minutes=max_duration_value,
    )


def filter_valid_finished_alerts(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Keep records that can contribute positive duration to a time series."""

    valid = dataframe[
        dataframe["started_at_utc"].notna()
        & dataframe["finished_at_utc"].notna()
        & dataframe["oblast_name"].notna()
        & (dataframe["finished_at_utc"] >= dataframe["started_at_utc"])
    ].copy()
    return valid


def split_alerts_by_frequency(
    dataframe: pd.DataFrame,
    *,
    frequency: str,
) -> pd.DataFrame:
    """Split alert durations across UTC day or hour boundaries."""

    if frequency not in {"D", "h"}:
        raise ValueError("frequency must be 'D' for daily or 'h' for hourly.")

    records: list[dict[str, Any]] = []
    for row in dataframe.itertuples(index=False):
        start = getattr(row, "started_at_utc")
        finish = getattr(row, "finished_at_utc")
        if pd.isna(start) or pd.isna(finish) or finish <= start:
            continue

        cursor = start
        while cursor < finish:
            next_boundary = cursor.floor(frequency) + pd.Timedelta(1, unit=frequency)
            segment_end = min(finish, next_boundary)
            segment_minutes = (segment_end - cursor).total_seconds() / 60

            if segment_minutes > 0:
                records.append(
                    {
                        "source_record_id": getattr(row, "source_record_id"),
                        "data_source": getattr(row, "data_source"),
                        "oblast_name": getattr(row, "oblast_name"),
                        "alert_type": getattr(row, "alert_type"),
                        "period_start_utc": cursor.floor(frequency),
                        "alert_minutes": segment_minutes,
                    }
                )
            cursor = segment_end

    return pd.DataFrame(
        records,
        columns=[
            "source_record_id",
            "data_source",
            "oblast_name",
            "alert_type",
            "period_start_utc",
            "alert_minutes",
        ],
    )


def build_daily_timeseries(valid_alerts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level alerts into oblast-level daily UTC series."""

    split = split_alerts_by_frequency(valid_alerts, frequency="D")
    if split.empty:
        return empty_daily_timeseries()

    event_counts = valid_alerts.copy()
    event_counts["date"] = event_counts["started_at_utc"].dt.floor("D")

    duration_daily = (
        split.groupby(["period_start_utc", "oblast_name"], dropna=False)
        .agg(total_alert_minutes=("alert_minutes", "sum"))
        .reset_index()
        .rename(columns={"period_start_utc": "date"})
    )

    counts_daily = (
        event_counts.groupby(["date", "oblast_name"], dropna=False)
        .agg(
            alert_count=("source_record_id", "nunique"),
            github_records_count=(
                "data_source",
                lambda values: int((values == DATA_SOURCE_COLUMNS[0]).sum()),
            ),
            api_records_count=(
                "data_source",
                lambda values: int((values == DATA_SOURCE_COLUMNS[1]).sum()),
            ),
        )
        .reset_index()
    )

    daily = duration_daily.merge(counts_daily, on=["date", "oblast_name"], how="left")
    daily["alert_count"] = daily["alert_count"].fillna(0).astype("int64")
    daily["github_records_count"] = daily["github_records_count"].fillna(0).astype("int64")
    daily["api_records_count"] = daily["api_records_count"].fillna(0).astype("int64")
    daily["average_alert_duration"] = daily["total_alert_minutes"] / daily["alert_count"]
    daily.loc[daily["alert_count"] == 0, "average_alert_duration"] = pd.NA
    daily["had_alert_binary"] = (daily["total_alert_minutes"] > 0).astype("int64")
    daily["date_kyiv"] = daily["date"].dt.tz_convert(KYIV_TIMEZONE).dt.date.astype("string")

    daily["date"] = daily["date"].dt.date.astype("string")
    return daily[
        [
            "date",
            "date_kyiv",
            "oblast_name",
            "alert_count",
            "total_alert_minutes",
            "average_alert_duration",
            "had_alert_binary",
            "github_records_count",
            "api_records_count",
        ]
    ].sort_values(["date", "oblast_name"])


def build_hourly_timeseries(valid_alerts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level alerts into oblast-level hourly UTC series."""

    split = split_alerts_by_frequency(valid_alerts, frequency="h")
    if split.empty:
        return empty_hourly_timeseries()

    event_counts = valid_alerts.copy()
    event_counts["hour_start_utc"] = event_counts["started_at_utc"].dt.floor("h")

    duration_hourly = (
        split.groupby(["period_start_utc", "oblast_name"], dropna=False)
        .agg(total_alert_minutes=("alert_minutes", "sum"))
        .reset_index()
        .rename(columns={"period_start_utc": "hour_start_utc"})
    )

    counts_hourly = (
        event_counts.groupby(["hour_start_utc", "oblast_name"], dropna=False)
        .agg(
            alert_count=("source_record_id", "nunique"),
            github_records_count=(
                "data_source",
                lambda values: int((values == DATA_SOURCE_COLUMNS[0]).sum()),
            ),
            api_records_count=(
                "data_source",
                lambda values: int((values == DATA_SOURCE_COLUMNS[1]).sum()),
            ),
        )
        .reset_index()
    )

    hourly = duration_hourly.merge(
        counts_hourly,
        on=["hour_start_utc", "oblast_name"],
        how="left",
    )
    hourly["alert_count"] = hourly["alert_count"].fillna(0).astype("int64")
    hourly["github_records_count"] = hourly["github_records_count"].fillna(0).astype("int64")
    hourly["api_records_count"] = hourly["api_records_count"].fillna(0).astype("int64")
    hourly["average_alert_duration"] = hourly["total_alert_minutes"] / hourly["alert_count"]
    hourly.loc[hourly["alert_count"] == 0, "average_alert_duration"] = pd.NA
    hourly["had_alert_binary"] = (hourly["total_alert_minutes"] > 0).astype("int64")
    hourly["hour_start_kyiv"] = (
        hourly["hour_start_utc"].dt.tz_convert(KYIV_TIMEZONE).astype("string")
    )
    hourly["date"] = hourly["hour_start_utc"].dt.date.astype("string")
    hourly["date_kyiv"] = (
        hourly["hour_start_utc"].dt.tz_convert(KYIV_TIMEZONE).dt.date.astype("string")
    )
    hourly["hour_start_utc"] = hourly["hour_start_utc"].astype("string")

    return hourly[
        [
            "hour_start_utc",
            "hour_start_kyiv",
            "date",
            "date_kyiv",
            "oblast_name",
            "alert_count",
            "total_alert_minutes",
            "average_alert_duration",
            "had_alert_binary",
            "github_records_count",
            "api_records_count",
        ]
    ].sort_values(["hour_start_utc", "oblast_name"])


def write_outputs(
    daily: pd.DataFrame,
    hourly: pd.DataFrame | None,
    report: dict[str, Any],
    *,
    daily_output_path: Path = DAILY_OUTPUT_PATH,
    hourly_output_path: Path = HOURLY_OUTPUT_PATH,
    report_path: Path = PREPROCESS_REPORT_PATH,
) -> None:
    """Save preprocessed time series and validation report."""

    daily_output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(daily_output_path, index=False)
    print(f"Saved daily oblast time series: {daily_output_path}")

    if hourly is not None:
        hourly.to_csv(hourly_output_path, index=False)
        print(f"Saved hourly oblast time series: {hourly_output_path}")

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved preprocessing report: {report_path}")


def build_report(
    validation: ValidationResult,
    valid_alerts: pd.DataFrame,
    daily: pd.DataFrame,
    hourly: pd.DataFrame | None,
) -> dict[str, Any]:
    """Build a JSON report for validation and output coverage."""

    return {
        "input_rows": validation.input_rows,
        "valid_finished_rows_used": int(len(valid_alerts)),
        "duplicate_event_records": validation.duplicate_event_records,
        "missing_started_at": validation.missing_started_at,
        "missing_finished_at": validation.missing_finished_at,
        "negative_duration_rows": validation.negative_duration_rows,
        "zero_duration_rows": validation.zero_duration_rows,
        "long_duration_rows_over_7_days": validation.long_duration_rows,
        "max_duration_minutes": validation.max_duration_minutes,
        "impossible_date_rows": validation.impossible_date_rows,
        "daily_rows": int(len(daily)),
        "hourly_rows": None if hourly is None else int(len(hourly)),
        "daily_date_range": date_range(daily, "date"),
        "hourly_date_range": None if hourly is None else date_range(hourly, "date"),
        "unique_oblasts_count": int(daily["oblast_name"].dropna().nunique()),
    }


def print_summary(
    daily: pd.DataFrame,
    hourly: pd.DataFrame | None,
    report: dict[str, Any],
    preview_rows: int,
) -> None:
    print("\nTime series preprocessing summary")
    print(f"- input rows: {report['input_rows']:,}")
    print(f"- valid finished rows used: {report['valid_finished_rows_used']:,}")
    print(f"- duplicate event records: {report['duplicate_event_records']:,}")
    print(f"- negative duration rows: {report['negative_duration_rows']:,}")
    print(f"- long duration rows over 7 days: {report['long_duration_rows_over_7_days']:,}")
    print(f"- daily rows: {report['daily_rows']:,}")
    print(f"- hourly rows: {report['hourly_rows']:,}" if hourly is not None else "- hourly rows: skipped")
    print(f"- unique oblasts: {report['unique_oblasts_count']:,}")
    print(f"- daily columns: {list(daily.columns)}")

    print(f"\nFirst {preview_rows} daily rows")
    if daily.empty:
        print("<no rows>")
    else:
        print(daily.head(preview_rows).to_string(index=False))

    if hourly is not None:
        print(f"\nFirst {preview_rows} hourly rows")
        if hourly.empty:
            print("<no rows>")
        else:
            print(hourly.head(preview_rows).to_string(index=False))


def require_columns(columns: pd.Index) -> None:
    missing = sorted(REQUIRED_MERGED_COLUMNS - set(columns))
    if missing:
        raise ValueError(f"Merged alert dataset missing required columns: {missing}")


def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")


def date_range(dataframe: pd.DataFrame, column: str) -> dict[str, str | None]:
    if dataframe.empty:
        return {"min": None, "max": None}
    return {
        "min": str(dataframe[column].min()),
        "max": str(dataframe[column].max()),
    }


def empty_daily_timeseries() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "date_kyiv",
            "oblast_name",
            "alert_count",
            "total_alert_minutes",
            "average_alert_duration",
            "had_alert_binary",
            "github_records_count",
            "api_records_count",
        ]
    )


def empty_hourly_timeseries() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "hour_start_utc",
            "hour_start_kyiv",
            "date",
            "date_kyiv",
            "oblast_name",
            "alert_count",
            "total_alert_minutes",
            "average_alert_duration",
            "had_alert_binary",
            "github_records_count",
            "api_records_count",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build oblast-level daily/hourly time series from merged alerts."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=MERGED_OUTPUT_PATH,
        help="Path to data/processed/alerts_merged_event_level.csv.",
    )
    parser.add_argument(
        "--skip-hourly",
        action="store_true",
        help="Only build the daily output.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Number of rows to print from each output. Default: 5.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    merged = load_merged_alerts(args.input_path)
    validation = validate_merged_alerts(merged)
    valid_alerts = filter_valid_finished_alerts(merged)

    daily = build_daily_timeseries(valid_alerts)
    hourly = None if args.skip_hourly else build_hourly_timeseries(valid_alerts)
    report = build_report(validation, valid_alerts, daily, hourly)

    write_outputs(daily, hourly, report)
    print_summary(daily, hourly, report, args.preview_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

