"""Fetch recent alerts.in.ua records as a delta layer.

The long-term baseline comes from the GitHub official CSV. This script reads
the newest GitHub timestamp, fetches a short overlapping window from the
alerts.in.ua history endpoint, and saves the normalized API-only delta. It does
not merge API data into the GitHub baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from api_client import (
    HISTORY_PERIOD_MONTH_AGO,
    AlertsInUaClient,
    AlertsInUaError,
    AlertsInUaRateLimitError,
)
from fetch_github_dataset import RAW_DATASET_PATH


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_API_DELTA_DIR = PROJECT_ROOT / "data" / "raw" / "api_delta"
PROCESSED_API_DELTA_PATH = PROJECT_ROOT / "data" / "processed" / "api_delta_alerts.csv"

DATA_SOURCE = "alerts_api"
DEFAULT_OVERLAP_DAYS = 7
DEFAULT_HISTORY_SLEEP_SECONDS = 31.0


@dataclass(frozen=True)
class DeltaWindow:
    """UTC time window used to filter API history."""

    github_max_at: pd.Timestamp
    api_start_at: pd.Timestamp
    api_end_at: pd.Timestamp


def ensure_directories() -> None:
    RAW_API_DELTA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_API_DELTA_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_github_max_started_at(raw_path: Path = RAW_DATASET_PATH) -> pd.Timestamp:
    """Read the newest started_at timestamp from the preserved GitHub raw CSV."""

    if not raw_path.exists():
        raise FileNotFoundError(
            f"GitHub raw dataset not found: {raw_path}. "
            "Run src/fetch_github_dataset.py first."
        )

    started_at = pd.read_csv(raw_path, usecols=["started_at"])["started_at"]
    parsed = pd.to_datetime(started_at, errors="coerce", utc=True).dropna()
    if parsed.empty:
        raise ValueError(f"No valid started_at timestamps found in {raw_path}.")

    return parsed.max()


def build_delta_window(
    *,
    raw_path: Path = RAW_DATASET_PATH,
    overlap_days: int = DEFAULT_OVERLAP_DAYS,
    api_end_at: pd.Timestamp | None = None,
) -> DeltaWindow:
    """Build the API delta window from the GitHub max timestamp."""

    github_max_at = read_github_max_started_at(raw_path)
    end_at = api_end_at or (
        pd.Timestamp.now(tz="UTC").normalize()
        + pd.Timedelta(days=1)
        - pd.Timedelta(microseconds=1)
    )
    return DeltaWindow(
        github_max_at=github_max_at,
        api_start_at=github_max_at - pd.Timedelta(days=overlap_days),
        api_end_at=end_at,
    )


def parse_region_selection(client: AlertsInUaClient, regions: str) -> list[dict[str, str]]:
    """Parse 'all' or a comma-separated UID list into documented region records."""

    locations = client.list_documented_locations()
    by_uid = {location["uid"]: location for location in locations}
    requested = regions.strip()

    if requested.casefold() == "all":
        return locations

    selected: list[dict[str, str]] = []
    for raw_uid in requested.split(","):
        uid = raw_uid.strip()
        if not uid:
            continue
        if not uid.isdigit():
            raise ValueError(f"Region UID must be numeric, got {uid!r}.")
        selected.append(by_uid.get(uid, {"uid": uid, "name": "", "type": "unknown"}))

    if not selected:
        raise ValueError("No regions selected.")
    return selected


def raw_response_path(uid: str, window: DeltaWindow, period: str) -> Path:
    """Build a deterministic cache path for one UID/window request."""

    start_label = window.api_start_at.strftime("%Y%m%dT%H%M%SZ")
    end_label = window.api_end_at.strftime("%Y%m%dT%H%M%SZ")
    return RAW_API_DELTA_DIR / f"alerts_api_uid_{uid}_{period}_{start_label}_{end_label}.json"


def fetch_or_load_region_history(
    *,
    client: AlertsInUaClient,
    region: dict[str, str],
    window: DeltaWindow,
    period: str,
    force: bool,
) -> dict[str, Any]:
    """Fetch one region's history response or reuse a saved raw JSON file."""

    ensure_directories()
    uid = region["uid"]
    path = raw_response_path(uid, window, period)

    if path.exists() and not force:
        print(f"Using cached API response for UID {uid}: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    print(f"Fetching alerts.in.ua history for UID {uid} ({region.get('name') or 'unknown'})")
    result = client.get_alert_history(uid, period=period)
    raw_payload = {
        "request": {
            "uid": uid,
            "region_name": region.get("name"),
            "region_type": region.get("type"),
            "period": period,
            "api_start_at": window.api_start_at.isoformat(),
            "api_end_at": window.api_end_at.isoformat(),
            "github_max_started_at": window.github_max_at.isoformat(),
        },
        "response_meta": {
            "status_code": result.meta.status_code,
            "url": result.meta.url,
            "content_type": result.meta.content_type,
            "last_modified": result.meta.last_modified,
            "retry_after": result.meta.retry_after,
        },
        "data": result.data,
    }

    path.write_text(
        json.dumps(raw_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved raw API response: {path}")
    return raw_payload


def normalize_api_payloads(
    payloads: list[dict[str, Any]],
    *,
    window: DeltaWindow,
) -> pd.DataFrame:
    """Convert raw API response payloads into one normalized delta dataframe."""

    rows: list[dict[str, Any]] = []
    for payload in payloads:
        request = payload.get("request", {})
        meta = payload.get("response_meta", {})
        data = payload.get("data", {})
        alerts = data.get("alerts", []) if isinstance(data, dict) else []

        for alert in alerts:
            if not isinstance(alert, dict):
                continue
            rows.append(normalize_alert(alert, request=request, meta=meta))

    dataframe = pd.DataFrame(rows, columns=normalized_columns())
    if dataframe.empty:
        return dataframe

    for column in ("started_at", "finished_at", "updated_at", "deleted_at"):
        dataframe[column] = pd.to_datetime(dataframe[column], errors="coerce", utc=True)

    started = dataframe["started_at"]
    in_window = started.notna() & (started >= window.api_start_at) & (started <= window.api_end_at)
    dataframe = dataframe.loc[in_window].copy()

    dataframe["duration_minutes"] = (
        dataframe["finished_at"] - dataframe["started_at"]
    ).dt.total_seconds() / 60
    dataframe.loc[dataframe["finished_at"].isna(), "duration_minutes"] = pd.NA

    dataframe = dataframe.drop_duplicates(
        subset=["source_id", "region_uid", "alert_type", "started_at"],
        keep="last",
    )
    return dataframe.sort_values(["started_at", "region_uid"], na_position="last")


def normalize_alert(
    alert: dict[str, Any],
    *,
    request: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Normalize one alerts.in.ua alert object."""

    return {
        "data_source": DATA_SOURCE,
        "source_id": alert.get("id"),
        "source_url": meta.get("url"),
        "requested_region_uid": request.get("uid"),
        "requested_region_name": request.get("region_name"),
        "requested_region_type": request.get("region_type"),
        "region_uid": as_nullable_string(alert.get("location_uid")),
        "region_name": alert.get("location_title"),
        "region_type": alert.get("location_type"),
        "oblast_uid": as_nullable_string(alert.get("location_oblast_uid")),
        "oblast_name": alert.get("location_oblast"),
        "raion_name": alert.get("location_raion"),
        "alert_type": alert.get("alert_type"),
        "started_at": alert.get("started_at"),
        "finished_at": alert.get("finished_at"),
        "updated_at": alert.get("updated_at"),
        "deleted_at": alert.get("deleted_at"),
        "duration_minutes": pd.NA,
        "notes": alert.get("notes"),
        "country": alert.get("country"),
        "calculated": alert.get("calculated"),
    }


def save_delta_dataframe(
    dataframe: pd.DataFrame,
    path: Path = PROCESSED_API_DELTA_PATH,
) -> Path:
    """Save the normalized API delta as CSV."""

    ensure_directories()
    dataframe.to_csv(path, index=False)
    print(f"Saved normalized API delta: {path}")
    return path


def normalized_columns() -> list[str]:
    return [
        "data_source",
        "source_id",
        "source_url",
        "requested_region_uid",
        "requested_region_name",
        "requested_region_type",
        "region_uid",
        "region_name",
        "region_type",
        "oblast_uid",
        "oblast_name",
        "raion_name",
        "alert_type",
        "started_at",
        "finished_at",
        "updated_at",
        "deleted_at",
        "duration_minutes",
        "notes",
        "country",
        "calculated",
    ]


def as_nullable_string(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def print_delta_summary(dataframe: pd.DataFrame, window: DeltaWindow, preview_rows: int) -> None:
    print("\nAPI delta window")
    print(f"- github_max_started_at: {window.github_max_at}")
    print(f"- api_start_date: {window.api_start_at}")
    print(f"- api_end_date: {window.api_end_at}")

    print("\nNormalized API delta")
    print(f"- rows: {len(dataframe):,}")
    print(f"- columns: {list(dataframe.columns)}")
    if dataframe.empty:
        print("- first rows: <no rows in overlap window>")
        return

    print("- date_range_utc:")
    print(f"  {dataframe['started_at'].min()} to {dataframe['started_at'].max()}")
    print("- region_names:")
    print(f"  {sorted(dataframe['region_name'].dropna().unique().tolist())[:20]}")
    print(f"\nFirst {preview_rows} rows")
    print(dataframe.head(preview_rows).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch recent alerts.in.ua history as a separate delta layer."
    )
    parser.add_argument(
        "--regions",
        default="31",
        help=(
            "Comma-separated documented alerts.in.ua UIDs, or 'all'. "
            "Default: 31 (Kyiv city) for a safe smoke run."
        ),
    )
    parser.add_argument(
        "--period",
        default=HISTORY_PERIOD_MONTH_AGO,
        help="alerts.in.ua history period. Public docs currently list 'month_ago'.",
    )
    parser.add_argument(
        "--overlap-days",
        type=int,
        default=DEFAULT_OVERLAP_DAYS,
        help="Days to subtract from the GitHub max started_at. Default: 7.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_HISTORY_SLEEP_SECONDS,
        help=(
            "Minimum delay between API requests. Default: 31 seconds for the "
            "documented history endpoint limit of 2 requests/minute."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refetch API responses even if matching raw JSON files already exist.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Number of normalized rows to print. Default: 5.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.period != HISTORY_PERIOD_MONTH_AGO:
        raise ValueError(
            f"Unsupported API history period {args.period!r}. "
            f"The current client supports {HISTORY_PERIOD_MONTH_AGO!r}."
        )

    ensure_directories()
    window = build_delta_window(overlap_days=args.overlap_days)
    client = AlertsInUaClient(min_interval_seconds=args.sleep_seconds)
    regions = parse_region_selection(client, args.regions)

    print("alerts.in.ua delta ingestion")
    print("Disclaimer: educational data ingestion only; not for operational use.")
    print(f"Selected regions: {[region['uid'] for region in regions]}")
    print(f"History request delay: {args.sleep_seconds}s")

    payloads: list[dict[str, Any]] = []
    for index, region in enumerate(regions, start=1):
        try:
            payloads.append(
                fetch_or_load_region_history(
                    client=client,
                    region=region,
                    window=window,
                    period=args.period,
                    force=args.force,
                )
            )
        except AlertsInUaRateLimitError as exc:
            print(f"Rate limit reached after {index - 1} successful request(s): {exc}")
            print("Stopping now to avoid spamming the API. Rerun later to continue.")
            break
        except AlertsInUaError as exc:
            print(f"API request failed for UID {region['uid']}: {exc}")
            return 1

    dataframe = normalize_api_payloads(payloads, window=window)
    save_delta_dataframe(dataframe)
    print_delta_summary(dataframe, window, args.preview_rows)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
