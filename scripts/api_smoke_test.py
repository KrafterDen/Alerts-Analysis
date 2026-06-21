"""Smoke test for the alerts.in.ua API exploration client.

Usage:
    python scripts/api_smoke_test.py
    python scripts/api_smoke_test.py --include-history --uid 31

Set ALERTS_IN_UA_TOKEN in .env before running.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from api_client import (  # noqa: E402
    HISTORY_LIMIT_REQUESTS_PER_MINUTE,
    TOKEN_ENV_VAR,
    AlertsInUaClient,
    AlertsInUaError,
    MissingTokenError,
    load_environment,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch a small alerts.in.ua sample and print response structure."
    )
    parser.add_argument(
        "--uid",
        default="31",
        help="Region UID for optional history/status checks. Default: 31, Kyiv city.",
    )
    parser.add_argument(
        "--include-history",
        action="store_true",
        help=(
            "Also call the history endpoint. The docs list a separate limit of "
            f"{HISTORY_LIMIT_REQUESTS_PER_MINUTE} requests/minute for this endpoint."
        ),
    )
    args = parser.parse_args()

    print("alerts.in.ua API smoke test")
    print(f"Token source: {TOKEN_ENV_VAR}")
    print("Disclaimer: educational data exploration only; not for operational use.")
    print()

    load_environment(PROJECT_ROOT / ".env")

    if not os.getenv(TOKEN_ENV_VAR):
        print(f"Missing token. Set {TOKEN_ENV_VAR} in .env and run this script again.")
        return 2

    try:
        client = AlertsInUaClient()

        print("Documented location source")
        locations = client.list_documented_locations()
        print(f"- No dedicated locations endpoint is documented.")
        print(f"- Using {len(locations)} documented oblast/special-city UIDs.")
        print(f"- First locations: {locations[:5]}")
        print()

        print("GET /v1/alerts/active.json")
        active_alerts = client.get_active_alerts()
        print_result(active_alerts.data, active_alerts.meta)
        print()

        print("GET /v1/iot/active_air_raid_alerts_by_oblast.json")
        oblast_statuses = client.get_air_raid_statuses_by_oblast()
        print_result(oblast_statuses.data, oblast_statuses.meta)
        print()

        print(f"GET /v1/iot/active_air_raid_alerts/{args.uid}.json")
        uid_status = client.get_air_raid_status_by_uid(args.uid)
        print_result(uid_status.data, uid_status.meta)
        print()

        if args.include_history:
            print(f"GET /v1/regions/{args.uid}/alerts/month_ago.json")
            history = client.get_alert_history(args.uid, period="month_ago")
            print_result(history.data, history.meta)
            print()
        else:
            print("Skipped history endpoint. Use --include-history to test it.")

    except MissingTokenError as exc:
        print(exc)
        return 2
    except AlertsInUaError as exc:
        print(f"API smoke test failed: {exc}")
        return 1

    return 0


def print_result(data: Any, meta: Any) -> None:
    print(f"- status_code: {meta.status_code}")
    print(f"- content_type: {meta.content_type}")
    print(f"- last_modified: {meta.last_modified}")
    print(f"- response_type: {type(data).__name__}")
    print("- structure:")
    print_structure(data, indent=2)


def print_structure(value: Any, *, indent: int = 0, max_list_items: int = 1) -> None:
    prefix = " " * indent

    if isinstance(value, dict):
        if not value:
            print(f"{prefix}{{}}")
            return
        print(f"{prefix}object with keys: {list(value.keys())}")
        for key, item in value.items():
            print(f"{prefix}- {key}: {type_name(item)}")
            if isinstance(item, (dict, list)):
                print_structure(item, indent=indent + 4, max_list_items=max_list_items)
        return

    if isinstance(value, list):
        print(f"{prefix}list length: {len(value)}")
        for index, item in enumerate(value[:max_list_items]):
            print(f"{prefix}- sample[{index}]: {type_name(item)}")
            print_structure(item, indent=indent + 4, max_list_items=max_list_items)
        return

    if isinstance(value, str):
        unique_symbols = "".join(sorted(set(value)))
        sample = value[:80]
        print(f"{prefix}string length: {len(value)}")
        print(f"{prefix}unique symbols: {unique_symbols!r}")
        print(f"{prefix}sample: {sample!r}")
        return

    print(f"{prefix}{value!r}")


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    return type(value).__name__


if __name__ == "__main__":
    raise SystemExit(main())
