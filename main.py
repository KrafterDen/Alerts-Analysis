"""Project command-line entry point."""

from __future__ import annotations

import argparse

from src.analyze import print_analysis_summary, run_analysis
from src.events import (
    EXPANDED_EVENTS_PATH,
    build_expanded_event_calendar,
    print_event_summary,
)
from src.forecast import print_forecast_summary, run_forecast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KSE air raid alerts analysis project CLI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze",
        help="Run basic statistical analysis and generate static outputs.",
    )
    analyze.add_argument(
        "--oblast",
        required=True,
        help='Selected oblast for figures, e.g. "Kyiv Oblast" or "Kyiv City".',
    )

    events = subparsers.add_parser(
        "events",
        help="Expand holiday/symbolic-date events into dated event windows.",
    )
    events.add_argument(
        "--window-days-before",
        type=int,
        default=3,
        help="Days before each event date to include. Default: 3.",
    )
    events.add_argument(
        "--window-days-after",
        type=int,
        default=3,
        help="Days after each event date to include. Default: 3.",
    )

    forecast = subparsers.add_parser(
        "forecast",
        help="Run exploratory next-day alert-risk baseline models.",
    )
    forecast.add_argument(
        "--model",
        choices=["logistic_regression", "random_forest"],
        default="logistic_regression",
        help="Simple sklearn model to train. Default: logistic_regression.",
    )
    forecast.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of latest dates used for the time-based test split.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "analyze":
        outputs = run_analysis(oblast=args.oblast)
        print_analysis_summary(outputs)
        return 0

    if args.command == "events":
        expanded = build_expanded_event_calendar(
            window_days_before=args.window_days_before,
            window_days_after=args.window_days_after,
        )
        print_event_summary(expanded, EXPANDED_EVENTS_PATH)
        return 0

    if args.command == "forecast":
        result = run_forecast(model_type=args.model, test_size=args.test_size)
        print_forecast_summary(result)
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
