"""Event calendar expansion for holiday and symbolic-date analysis.

This module creates neutral analytical event windows. It supports comparison,
association, and uplift calculations later, but it does not claim causation.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "data" / "events" / "holiday_events.csv"
DAILY_TIMESERIES_PATH = PROJECT_ROOT / "data" / "processed" / "daily_oblast_timeseries.csv"
EXPANDED_EVENTS_PATH = PROJECT_ROOT / "data" / "processed" / "expanded_event_calendar.csv"

EVENT_CATEGORIES = {
    "ukrainian_public",
    "ukrainian_memorial",
    "russian_state",
    "russian_military",
    "shared_public",
    "other_symbolic",
}

REQUIRED_EVENT_COLUMNS = [
    "event_id",
    "name",
    "month",
    "day",
    "category",
    "country_or_context",
    "display_color",
    "notes",
]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class CoveredYears:
    """Year coverage from the daily time-series input."""

    start_year: int
    end_year: int
    min_date: date
    max_date: date


def load_holiday_events(path: Path = EVENTS_PATH) -> pd.DataFrame:
    """Load and validate the holiday/symbolic-date source dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Missing event dataset: {path}")

    dataframe = pd.read_csv(path, dtype="string")
    validate_event_dataset(dataframe)
    dataframe["month"] = pd.to_numeric(dataframe["month"], errors="coerce").astype("Int64")
    dataframe["day"] = pd.to_numeric(dataframe["day"], errors="coerce").astype("Int64")
    return dataframe


def get_covered_years(daily_path: Path = DAILY_TIMESERIES_PATH) -> CoveredYears:
    """Read the year coverage from daily_oblast_timeseries.csv."""

    if not daily_path.exists():
        raise FileNotFoundError(
            f"Missing daily time series: {daily_path}. "
            "Run src/preprocess_timeseries.py first."
        )

    dates = pd.to_datetime(
        pd.read_csv(daily_path, usecols=["date"])["date"],
        errors="coerce",
    ).dropna()
    if dates.empty:
        raise ValueError(f"No valid dates found in {daily_path}.")

    min_date = dates.min().date()
    max_date = dates.max().date()
    return CoveredYears(
        start_year=min_date.year,
        end_year=max_date.year,
        min_date=min_date,
        max_date=max_date,
    )


def expand_events(
    events: pd.DataFrame,
    covered_years: CoveredYears,
    *,
    window_days_before: int = 3,
    window_days_after: int = 3,
) -> pd.DataFrame:
    """Expand recurring event definitions into concrete dated event windows."""

    rows: list[dict[str, object]] = []
    for year in range(covered_years.start_year, covered_years.end_year + 1):
        for event in events.itertuples(index=False):
            event_date = concrete_event_date(event, year)
            window_start = event_date - timedelta(days=window_days_before)
            window_end = event_date + timedelta(days=window_days_after)

            rows.append(
                {
                    "event_instance_id": f"{event.event_id}_{year}",
                    "event_id": event.event_id,
                    "name": event.name,
                    "year": year,
                    "event_date": event_date.isoformat(),
                    "month": event_date.month,
                    "day": event_date.day,
                    "category": event.category,
                    "country_or_context": event.country_or_context,
                    "display_color": event.display_color,
                    "notes": event.notes,
                    "is_moving_date": bool(pd.isna(event.month) or pd.isna(event.day)),
                    "window_days_before": window_days_before,
                    "window_days_after": window_days_after,
                    "window_start_date": window_start.isoformat(),
                    "window_end_date": window_end.isoformat(),
                    "daily_data_min_date": covered_years.min_date.isoformat(),
                    "daily_data_max_date": covered_years.max_date.isoformat(),
                }
            )

    expanded = pd.DataFrame(rows)
    return expanded.sort_values(["event_date", "event_id"]).reset_index(drop=True)


def concrete_event_date(event: object, year: int) -> date:
    """Return a concrete date for a fixed-date or moving-date event."""

    event_id = str(getattr(event, "event_id"))
    month = getattr(event, "month")
    day = getattr(event, "day")

    if pd.notna(month) and pd.notna(day):
        return date(year, int(month), int(day))

    if event_id == "orthodox_easter":
        return orthodox_easter_date(year)

    raise ValueError(
        f"Event {event_id!r} is missing month/day and has no expansion rule."
    )


def orthodox_easter_date(year: int) -> date:
    """Calculate Orthodox Easter date in the Gregorian calendar.

    This uses the Julian-calendar computus and converts to Gregorian dates for
    1900-2099 by adding 13 days, which covers the current dataset range.
    """

    if not 1900 <= year <= 2099:
        raise ValueError(
            "orthodox_easter_date currently supports years 1900-2099."
        )

    a = year % 4
    b = year % 7
    c = year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    julian_month = (d + e + 114) // 31
    julian_day = ((d + e + 114) % 31) + 1
    julian_date = date(year, julian_month, julian_day)
    return julian_date + timedelta(days=13)


def save_expanded_events(
    expanded: pd.DataFrame,
    path: Path = EXPANDED_EVENTS_PATH,
) -> Path:
    """Save the expanded event calendar."""

    path.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_csv(path, index=False)
    return path


def build_expanded_event_calendar(
    *,
    events_path: Path = EVENTS_PATH,
    daily_path: Path = DAILY_TIMESERIES_PATH,
    output_path: Path = EXPANDED_EVENTS_PATH,
    window_days_before: int = 3,
    window_days_after: int = 3,
) -> pd.DataFrame:
    """Load, expand, and save the event calendar."""

    events = load_holiday_events(events_path)
    covered_years = get_covered_years(daily_path)
    expanded = expand_events(
        events,
        covered_years,
        window_days_before=window_days_before,
        window_days_after=window_days_after,
    )
    save_expanded_events(expanded, output_path)
    return expanded


def validate_event_dataset(dataframe: pd.DataFrame) -> None:
    """Validate source event schema and neutral category values."""

    missing = sorted(set(REQUIRED_EVENT_COLUMNS) - set(dataframe.columns))
    if missing:
        raise ValueError(f"holiday_events.csv missing required columns: {missing}")

    duplicated = dataframe["event_id"].duplicated().sum()
    if duplicated:
        raise ValueError(f"holiday_events.csv has {duplicated} duplicate event_id rows.")

    invalid_categories = sorted(set(dataframe["category"].dropna()) - EVENT_CATEGORIES)
    if invalid_categories:
        raise ValueError(f"Invalid event categories: {invalid_categories}")

    invalid_colors = [
        value
        for value in dataframe["display_color"].dropna()
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", str(value))
    ]
    if invalid_colors:
        raise ValueError(f"Invalid display colors: {invalid_colors}")

    for row in dataframe.itertuples(index=False):
        month = pd.to_numeric(getattr(row, "month"), errors="coerce")
        day = pd.to_numeric(getattr(row, "day"), errors="coerce")
        if pd.notna(month) or pd.notna(day):
            if pd.isna(month) or pd.isna(day):
                raise ValueError(
                    f"Event {row.event_id!r} must define both month and day or neither."
                )
            date(2024, int(month), int(day))


def print_event_summary(expanded: pd.DataFrame, path: Path) -> None:
    """Print a compact CLI summary."""

    print("Expanded event calendar created")
    print("Neutral framing: association, comparison, event window, and uplift only.")
    print(f"Rows: {len(expanded):,}")
    print(f"Years: {expanded['year'].min()}-{expanded['year'].max()}")
    print(f"Categories: {sorted(expanded['category'].unique().tolist())}")
    print(f"Saved: {path}")
    print("\nFirst rows")
    print(expanded.head(10).to_string(index=False))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand holiday and symbolic-date events into dated event windows."
    )
    parser.add_argument(
        "--window-days-before",
        type=int,
        default=3,
        help="Days before each event date to include in the event window. Default: 3.",
    )
    parser.add_argument(
        "--window-days-after",
        type=int,
        default=3,
        help="Days after each event date to include in the event window. Default: 3.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    expanded = build_expanded_event_calendar(
        window_days_before=args.window_days_before,
        window_days_after=args.window_days_after,
    )
    print_event_summary(expanded, EXPANDED_EVENTS_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

