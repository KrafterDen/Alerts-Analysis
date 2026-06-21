"""Project command-line entry point."""

from __future__ import annotations

import argparse

from src.analyze import print_analysis_summary, run_analysis


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

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "analyze":
        outputs = run_analysis(oblast=args.oblast)
        print_analysis_summary(outputs)
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

