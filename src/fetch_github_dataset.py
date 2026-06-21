"""Fetch and inspect the official historical air-raid alerts dataset.

The historical source is:
https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset

This module only ingests and lightly normalizes the official CSV. It does not
perform analysis, forecasting, or visualization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

OFFICIAL_DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_en.csv"
)
RAW_DATASET_PATH = RAW_DIR / "github_official_alerts.csv"
PROCESSED_DATASET_PATH = PROCESSED_DIR / "github_official_alerts_processed.csv"
DATA_SOURCE = "github_official"

REQUIRED_COLUMNS = {
    "oblast",
    "raion",
    "hromada",
    "level",
    "started_at",
    "finished_at",
    "source",
}
TIMESTAMP_COLUMNS = ("started_at", "finished_at")


@dataclass(frozen=True)
class DatasetSummary:
    """Compact metadata for CLI reporting."""

    columns: list[str]
    date_min: pd.Timestamp | None
    date_max: pd.Timestamp | None
    row_count: int
    oblast_names: list[str]
    location_names: list[str]
    level_counts: dict[str, int]


def ensure_project_directories() -> None:
    """Create the data/output folders expected by the project."""

    for directory in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def download_official_dataset(
    *,
    source_url: str = OFFICIAL_DATASET_URL,
    raw_path: Path = RAW_DATASET_PATH,
    force: bool = False,
    timeout_seconds: float = 30.0,
) -> Path:
    """Download the official CSV, preserving the raw file contents."""

    ensure_project_directories()

    if raw_path.exists() and not force:
        print(f"Using cached raw dataset: {raw_path}")
        return raw_path

    print(f"Downloading official dataset from: {source_url}")
    response = requests.get(
        source_url,
        headers={"User-Agent": "kse-alerts-timeseries-ingestion/0.1"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "text" not in content_type and "csv" not in content_type:
        raise ValueError(
            f"Unexpected content type for CSV download: {content_type!r}"
        )

    temporary_path = raw_path.with_suffix(raw_path.suffix + ".tmp")
    temporary_path.write_bytes(response.content)
    temporary_path.replace(raw_path)
    print(f"Saved raw dataset: {raw_path}")
    return raw_path


def load_raw_dataset(raw_path: Path = RAW_DATASET_PATH) -> pd.DataFrame:
    """Load the preserved raw CSV and parse GitHub timestamps as UTC."""

    dataframe = pd.read_csv(raw_path, dtype="string")
    validate_columns(dataframe.columns)

    for column in TIMESTAMP_COLUMNS:
        dataframe[column] = pd.to_datetime(
            dataframe[column],
            errors="coerce",
            utc=True,
        )

    return dataframe


def build_processed_dataset(
    raw_dataframe: pd.DataFrame,
    *,
    source_url: str = OFFICIAL_DATASET_URL,
) -> pd.DataFrame:
    """Create a light processed copy while preserving source location levels."""

    dataframe = raw_dataframe.copy()
    validate_columns(dataframe.columns)

    dataframe["oblast_name"] = dataframe["oblast"]
    dataframe["location_level"] = dataframe["level"]
    dataframe["location_name"] = dataframe.apply(build_location_name, axis=1)

    duration = dataframe["finished_at"] - dataframe["started_at"]
    dataframe["duration_minutes"] = duration.dt.total_seconds() / 60
    dataframe.loc[dataframe["finished_at"].isna(), "duration_minutes"] = pd.NA

    dataframe["data_source"] = DATA_SOURCE
    dataframe["source_url"] = source_url

    ordered_columns = [
        "data_source",
        "source",
        "source_url",
        "oblast",
        "oblast_name",
        "raion",
        "hromada",
        "level",
        "location_level",
        "location_name",
        "started_at",
        "finished_at",
        "duration_minutes",
    ]
    return dataframe[ordered_columns]


def save_processed_dataset(
    dataframe: pd.DataFrame,
    processed_path: Path = PROCESSED_DATASET_PATH,
) -> Path:
    """Save the processed dataset as CSV."""

    ensure_project_directories()
    dataframe.to_csv(processed_path, index=False)
    print(f"Saved processed dataset: {processed_path}")
    return processed_path


def summarize_dataset(dataframe: pd.DataFrame) -> DatasetSummary:
    """Build summary fields requested for ingestion verification."""

    started_at = dataframe["started_at"].dropna()
    date_min = started_at.min() if not started_at.empty else None
    date_max = started_at.max() if not started_at.empty else None

    return DatasetSummary(
        columns=list(dataframe.columns),
        date_min=date_min,
        date_max=date_max,
        row_count=len(dataframe),
        oblast_names=unique_non_null_values(dataframe["oblast"]),
        location_names=unique_non_null_values(dataframe["location_name"]),
        level_counts=dataframe["level"].value_counts(dropna=False).to_dict(),
    )


def filter_by_oblast(dataframe: pd.DataFrame, oblast: str | None) -> pd.DataFrame:
    """Return a sample filtered to one oblast name, if requested."""

    if not oblast:
        return dataframe

    mask = dataframe["oblast"].str.casefold() == oblast.casefold()
    return dataframe.loc[mask].copy()


def print_summary(
    summary: DatasetSummary,
    *,
    max_locations: int = 30,
) -> None:
    """Print dataset structure and compact location metadata."""

    print("\nDataset summary")
    print(f"- columns: {summary.columns}")
    print(f"- date_range_utc: {summary.date_min} to {summary.date_max}")
    print(f"- rows: {summary.row_count:,}")
    print(f"- alert levels: {summary.level_counts}")
    print(f"- unique oblasts ({len(summary.oblast_names)}): {summary.oblast_names}")

    shown_locations = summary.location_names[:max_locations]
    print(f"- unique region/location names: {len(summary.location_names):,}")
    print(f"- first {len(shown_locations)} locations: {shown_locations}")


def print_preview(dataframe: pd.DataFrame, *, rows: int = 5) -> None:
    """Print columns and first rows for the requested sample."""

    print("\nPreview dataframe columns")
    print(list(dataframe.columns))
    print(f"\nFirst {rows} rows")
    if dataframe.empty:
        print("<no rows>")
        return
    print(dataframe.head(rows).to_string(index=False))


def validate_columns(columns: Iterable[str]) -> None:
    """Fail early if the upstream CSV schema changes."""

    missing = sorted(REQUIRED_COLUMNS - set(columns))
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")


def build_location_name(row: pd.Series) -> str:
    """Use the most specific available location while retaining oblast context."""

    for column in ("hromada", "raion", "oblast"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return "Unknown location"


def unique_non_null_values(series: pd.Series) -> list[str]:
    """Return sorted unique string values without missing entries."""

    values = {
        str(value).strip()
        for value in series.dropna()
        if str(value).strip()
    }
    return sorted(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and lightly process the official GitHub historical "
            "air-raid alerts dataset."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the raw CSV even if a cached copy exists.",
    )
    parser.add_argument(
        "--oblast",
        help="Optional exact oblast name for preview filtering, e.g. 'Kyivska oblast'.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Number of preview rows to print. Default: 5.",
    )
    parser.add_argument(
        "--max-locations",
        type=int,
        default=30,
        help="Maximum unique location names to print. Default: 30.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    raw_path = download_official_dataset(force=args.force)
    raw_dataframe = load_raw_dataset(raw_path)
    processed_dataframe = build_processed_dataset(raw_dataframe)
    save_processed_dataset(processed_dataframe)

    print_summary(
        summarize_dataset(processed_dataframe),
        max_locations=args.max_locations,
    )

    preview_dataframe = filter_by_oblast(processed_dataframe, args.oblast)
    if args.oblast:
        print(f"\nPreview filter: oblast == {args.oblast!r}")
        print(f"Filtered rows: {len(preview_dataframe):,}")
    print_preview(preview_dataframe, rows=args.preview_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

