"""Merge normalized GitHub historical data and alerts.in.ua API delta data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from fetch_github_dataset import RAW_DATASET_PATH
    from normalize import (
        API_DATA_SOURCE,
        API_DELTA_PATH,
        GITHUB_DATA_SOURCE,
        UNIFIED_COLUMNS,
        normalize_api_delta,
        normalize_github_alerts,
    )
except ImportError:
    from .fetch_github_dataset import RAW_DATASET_PATH
    from .normalize import (
        API_DATA_SOURCE,
        API_DELTA_PATH,
        GITHUB_DATA_SOURCE,
        UNIFIED_COLUMNS,
        normalize_api_delta,
        normalize_github_alerts,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MERGED_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "alerts_merged_event_level.csv"
MERGE_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "source_merge_report.json"

DEDUPLICATION_COLUMNS = [
    "oblast_name",
    "location_type",
    "alert_type",
    "started_at_key",
    "finished_at_key",
]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def merge_sources(
    *,
    github_raw_path: Path = RAW_DATASET_PATH,
    api_delta_path: Path = API_DELTA_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize, merge, deduplicate, and report source coverage."""

    github = normalize_github_alerts(github_raw_path)
    api = normalize_api_delta(api_delta_path)

    github_max_started_at = github["started_at_utc"].max()
    combined = pd.concat([github, api], ignore_index=True)
    combined = combined[UNIFIED_COLUMNS].copy()

    exact_duplicate_count = int(combined.duplicated(subset=UNIFIED_COLUMNS).sum())
    unique_combined = combined.drop_duplicates(subset=UNIFIED_COLUMNS, keep="first")

    deduplicated = deduplicate_alerts(unique_combined, github_max_started_at)
    deduplicated = deduplicated.sort_values(
        ["started_at_utc", "oblast_name", "location_type", "source_record_id"],
        na_position="last",
    ).reset_index(drop=True)
    cross_source_duplicate_count = len(unique_combined) - len(deduplicated)
    duplicate_count = exact_duplicate_count + cross_source_duplicate_count

    report = build_merge_report(
        github=github,
        api=api,
        merged=deduplicated,
        duplicate_rows_removed=duplicate_count,
        exact_duplicate_rows_removed=exact_duplicate_count,
        cross_source_duplicate_rows_removed=cross_source_duplicate_count,
        github_max_started_at=github_max_started_at,
    )
    return deduplicated, report


def deduplicate_alerts(
    dataframe: pd.DataFrame,
    github_max_started_at: pd.Timestamp,
) -> pd.DataFrame:
    """Remove cross-source overlap duplicates while preserving same-source rows."""

    working = dataframe.copy()
    working["started_at_key"] = round_timestamp_for_key(working["started_at_utc"])
    working["finished_at_key"] = round_timestamp_for_key(working["finished_at_utc"])
    working["_source_priority"] = source_preference_rank(working, github_max_started_at)
    working["_row_order"] = range(len(working))
    working["_keep_row"] = True

    for _, group in working.groupby(DEDUPLICATION_COLUMNS, dropna=False):
        if group["data_source"].nunique(dropna=True) <= 1:
            continue

        preferred_priority = group["_source_priority"].min()
        preferred_sources = set(
            group.loc[group["_source_priority"] == preferred_priority, "data_source"]
        )
        keep_mask = group["data_source"].isin(preferred_sources)
        working.loc[group.index, "_keep_row"] = keep_mask

    deduplicated = working.loc[working["_keep_row"]].sort_values("_row_order")
    return deduplicated[UNIFIED_COLUMNS].copy()


def source_preference_rank(
    dataframe: pd.DataFrame,
    github_max_started_at: pd.Timestamp,
) -> pd.Series:
    """Rank rows so deduplication keeps the preferred source first."""

    # Default priority prefers the long-term official GitHub source for finalized
    # records already covered by the GitHub data window.
    priority = pd.Series(2, index=dataframe.index)

    older_finalized_github = (
        (dataframe["data_source"] == GITHUB_DATA_SOURCE)
        & dataframe["is_finished"]
        & (dataframe["started_at_utc"] <= github_max_started_at)
    )
    recent_api = (
        (dataframe["data_source"] == API_DATA_SOURCE)
        & (dataframe["started_at_utc"] > github_max_started_at)
    )
    fallback_github = dataframe["data_source"] == GITHUB_DATA_SOURCE
    fallback_api = dataframe["data_source"] == API_DATA_SOURCE

    priority.loc[fallback_api] = 3
    priority.loc[fallback_github] = 1
    priority.loc[older_finalized_github] = 0
    priority.loc[recent_api] = 0
    return priority


def round_timestamp_for_key(series: pd.Series) -> pd.Series:
    """Round timestamps to the nearest minute for cross-source deduplication."""

    return pd.to_datetime(series, errors="coerce", utc=True).dt.round("min")


def build_merge_report(
    *,
    github: pd.DataFrame,
    api: pd.DataFrame,
    merged: pd.DataFrame,
    duplicate_rows_removed: int,
    exact_duplicate_rows_removed: int,
    cross_source_duplicate_rows_removed: int,
    github_max_started_at: pd.Timestamp,
) -> dict[str, Any]:
    """Build a JSON-serializable merge report."""

    return {
        "github_rows_count": int(len(github)),
        "api_rows_count": int(len(api)),
        "merged_rows_count": int(len(merged)),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "exact_duplicate_rows_removed": int(exact_duplicate_rows_removed),
        "cross_source_duplicate_rows_removed": int(cross_source_duplicate_rows_removed),
        "deduplication_key": [
            "oblast_name",
            "location_type",
            "alert_type",
            "started_at_utc rounded to minute",
            "finished_at_utc rounded to minute",
        ],
        "source_preference": {
            "older_finalized_data": GITHUB_DATA_SOURCE,
            "records_after_github_max_started_at": API_DATA_SOURCE,
            "github_max_started_at": timestamp_to_json(github_max_started_at),
        },
        "date_range_by_source": {
            GITHUB_DATA_SOURCE: date_range(github),
            API_DATA_SOURCE: date_range(api),
            "merged": date_range(merged),
        },
        "missing_finished_at_count": {
            GITHUB_DATA_SOURCE: int(github["finished_at_utc"].isna().sum()),
            API_DATA_SOURCE: int(api["finished_at_utc"].isna().sum()),
            "merged": int(merged["finished_at_utc"].isna().sum()),
        },
        "unique_oblasts_count": {
            GITHUB_DATA_SOURCE: int(github["oblast_name"].dropna().nunique()),
            API_DATA_SOURCE: int(api["oblast_name"].dropna().nunique()),
            "merged": int(merged["oblast_name"].dropna().nunique()),
        },
    }


def date_range(dataframe: pd.DataFrame) -> dict[str, str | None]:
    """Return min/max started_at timestamps for one source."""

    started = dataframe["started_at_utc"].dropna()
    if started.empty:
        return {"min_started_at_utc": None, "max_started_at_utc": None}
    return {
        "min_started_at_utc": timestamp_to_json(started.min()),
        "max_started_at_utc": timestamp_to_json(started.max()),
    }


def timestamp_to_json(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def save_outputs(
    merged: pd.DataFrame,
    report: dict[str, Any],
    *,
    merged_output_path: Path = MERGED_OUTPUT_PATH,
    report_path: Path = MERGE_REPORT_PATH,
) -> None:
    """Write merged CSV and JSON merge report."""

    merged_output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_output_path, index=False)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved merged event-level alerts: {merged_output_path}")
    print(f"Saved source merge report: {report_path}")


def print_merge_summary(
    merged: pd.DataFrame,
    report: dict[str, Any],
    preview_rows: int,
) -> None:
    print("\nMerge report summary")
    print(f"- GitHub rows: {report['github_rows_count']:,}")
    print(f"- API rows: {report['api_rows_count']:,}")
    print(f"- merged rows: {report['merged_rows_count']:,}")
    print(f"- duplicate rows removed: {report['duplicate_rows_removed']:,}")
    print(f"- unique oblasts: {report['unique_oblasts_count']['merged']:,}")
    print(f"- missing finished_at: {report['missing_finished_at_count']['merged']:,}")
    print(f"- columns: {list(merged.columns)}")

    print(f"\nFirst {preview_rows} merged rows")
    if merged.empty:
        print("<no rows>")
        return
    print(merged.head(preview_rows).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize and merge GitHub official alerts with API delta alerts."
    )
    parser.add_argument(
        "--github-raw-path",
        type=Path,
        default=RAW_DATASET_PATH,
        help="Path to data/raw/github_official_alerts.csv.",
    )
    parser.add_argument(
        "--api-delta-path",
        type=Path,
        default=API_DELTA_PATH,
        help="Path to data/processed/api_delta_alerts.csv.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Number of merged rows to print. Default: 5.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    merged, report = merge_sources(
        github_raw_path=args.github_raw_path,
        api_delta_path=args.api_delta_path,
    )
    save_outputs(merged, report)
    print_merge_summary(merged, report, args.preview_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
