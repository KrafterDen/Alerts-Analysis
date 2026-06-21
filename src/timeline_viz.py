"""Interactive Plotly holiday/event timeline visualization.

This is the project key visualization for comparing historical alert activity
and exploratory next-day alert-risk estimates around event windows. It is
analytical only and does not claim causation.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

try:
    from analyze import resolve_oblast_name
except ImportError:
    from .analyze import resolve_oblast_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAILY_TIMESERIES_PATH = PROJECT_ROOT / "data" / "processed" / "daily_oblast_timeseries.csv"
FORECAST_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_daily_oblast.csv"
EVENT_CALENDAR_PATH = PROJECT_ROOT / "data" / "processed" / "expanded_event_calendar.csv"
TIMELINE_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "holiday_timeline_preview.html"

DEFAULT_OBLAST = "Kyiv Oblast"
DEFAULT_ACTIVITY_METRIC = "alert_count"

CATEGORY_COLORS = {
    "russian_state": "#dc2626",
    "russian_military": "#b91c1c",
    "ukrainian_public": "#2563eb",
    "ukrainian_memorial": "#1d4ed8",
    "shared_public": "#64748b",
    "other_symbolic": "#111827",
    "multiple": "#7c3aed",
}

CATEGORY_LABELS = {
    "russian_state": "Russian state",
    "russian_military": "Russian military",
    "ukrainian_public": "Ukrainian public",
    "ukrainian_memorial": "Ukrainian memorial",
    "shared_public": "Shared public",
    "other_symbolic": "Other symbolic",
    "multiple": "Multiple events",
}

ACTIVITY_LABELS = {
    "alert_count": "Daily alert count",
    "total_alert_minutes": "Daily alert minutes",
}

EVENT_LABEL_LANES = 4
TIMELINE_MIN_WIDTH_PX = 4200
TIMELINE_MAX_WIDTH_PX = 9600
TIMELINE_PIXELS_PER_DAY = 5


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class TimelineData:
    """Prepared one-oblast timeline tables."""

    selected_oblast: str
    timeline: pd.DataFrame
    events: pd.DataFrame
    activity_metric: str


def load_daily_timeseries(path: Path = DAILY_TIMESERIES_PATH) -> pd.DataFrame:
    """Load daily actual alert activity."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing daily time series: {path}. Run src/preprocess_timeseries.py first."
        )

    dataframe = pd.read_csv(path)
    required = {"date", "oblast_name", "alert_count", "total_alert_minutes"}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Daily time series missing required columns: {missing}")

    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    dataframe["alert_count"] = pd.to_numeric(dataframe["alert_count"], errors="coerce")
    dataframe["total_alert_minutes"] = pd.to_numeric(
        dataframe["total_alert_minutes"],
        errors="coerce",
    )
    return dataframe.dropna(subset=["date", "oblast_name"])


def load_forecast(path: Path = FORECAST_PATH) -> pd.DataFrame:
    """Load forecast probabilities."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing forecast output: {path}. Run python main.py forecast first."
        )

    dataframe = pd.read_csv(path)
    required = {"forecast_date", "oblast_name", "model_probability", "split"}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Forecast output missing required columns: {missing}")

    dataframe["forecast_date"] = pd.to_datetime(
        dataframe["forecast_date"],
        errors="coerce",
    )
    dataframe["model_probability"] = pd.to_numeric(
        dataframe["model_probability"],
        errors="coerce",
    )
    return dataframe.dropna(subset=["forecast_date", "oblast_name"])


def load_events(path: Path = EVENT_CALENDAR_PATH) -> pd.DataFrame:
    """Load expanded event calendar."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing expanded event calendar: {path}. Run python main.py events first."
        )

    dataframe = pd.read_csv(path)
    required = {
        "name",
        "event_date",
        "category",
        "window_start_date",
        "window_end_date",
    }
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Expanded event calendar missing required columns: {missing}")

    dataframe["event_date"] = pd.to_datetime(dataframe["event_date"], errors="coerce")
    dataframe["window_start_date"] = pd.to_datetime(
        dataframe["window_start_date"],
        errors="coerce",
    )
    dataframe["window_end_date"] = pd.to_datetime(
        dataframe["window_end_date"],
        errors="coerce",
    )
    dataframe["marker_color"] = dataframe["category"].map(CATEGORY_COLORS).fillna(
        "#475569"
    )
    dataframe["category_label"] = dataframe["category"].map(CATEGORY_LABELS).fillna(
        dataframe["category"]
    )
    return dataframe.dropna(subset=["event_date", "window_start_date", "window_end_date"])


def prepare_timeline_data(
    *,
    oblast: str = DEFAULT_OBLAST,
    activity_metric: str = DEFAULT_ACTIVITY_METRIC,
    daily_path: Path = DAILY_TIMESERIES_PATH,
    forecast_path: Path = FORECAST_PATH,
    event_path: Path = EVENT_CALENDAR_PATH,
) -> TimelineData:
    """Prepare a selected-oblast timeline dataframe and event table."""

    if activity_metric not in ACTIVITY_LABELS:
        raise ValueError(
            f"activity_metric must be one of: {sorted(ACTIVITY_LABELS)}"
        )

    daily = load_daily_timeseries(daily_path)
    forecast = load_forecast(forecast_path)
    events = load_events(event_path)

    available_oblasts = sorted(
        set(daily["oblast_name"].dropna()) | set(forecast["oblast_name"].dropna())
    )
    selected_oblast = resolve_oblast_name(oblast, available_oblasts)

    selected_forecast = forecast[forecast["oblast_name"] == selected_oblast].copy()
    selected_daily = daily[daily["oblast_name"] == selected_oblast].copy()

    if selected_forecast.empty:
        raise ValueError(f"No forecast rows found for {selected_oblast}.")

    date_min = min(selected_forecast["forecast_date"].min(), selected_daily["date"].min())
    date_max = selected_forecast["forecast_date"].max()
    dates = pd.date_range(date_min, date_max, freq="D")

    timeline = pd.DataFrame({"date": dates})
    timeline["oblast_name"] = selected_oblast
    timeline = timeline.merge(
        selected_daily[["date", "alert_count", "total_alert_minutes"]],
        on="date",
        how="left",
    )
    timeline["has_actual_data"] = timeline["alert_count"].notna()
    timeline[["alert_count", "total_alert_minutes"]] = timeline[
        ["alert_count", "total_alert_minutes"]
    ].fillna(0)
    timeline = timeline.merge(
        selected_forecast[
            [
                "forecast_date",
                "model_probability",
                "rolling_7d_probability",
                "same_day_of_week_probability",
                "split",
            ]
        ].rename(columns={"forecast_date": "date"}),
        on="date",
        how="left",
    )
    timeline = add_event_hover_columns(timeline, events)
    timeline["model_probability_label"] = timeline["model_probability"].map(
        format_probability
    )
    return TimelineData(
        selected_oblast=selected_oblast,
        timeline=timeline,
        events=events,
        activity_metric=activity_metric,
    )


def add_event_hover_columns(timeline: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach exact-event and window-event labels for hover tooltips."""

    exact_lookup = build_exact_event_lookup(events)
    window_lookup = build_window_event_lookup(events)

    enriched = timeline.copy()
    enriched["event_name"] = enriched["date"].map(exact_lookup).fillna("")
    enriched["event_window_names"] = enriched["date"].map(window_lookup).fillna("")
    enriched["hover_event_text"] = enriched.apply(hover_event_text, axis=1)
    return enriched


def build_exact_event_lookup(events: pd.DataFrame) -> dict[pd.Timestamp, str]:
    """Map event dates to concise event labels."""

    lookup: dict[pd.Timestamp, list[str]] = {}
    for row in events.itertuples(index=False):
        day = pd.Timestamp(row.event_date).normalize()
        lookup.setdefault(day, []).append(str(row.name))
    return {day: "; ".join(names) for day, names in lookup.items()}


def build_window_event_lookup(events: pd.DataFrame) -> dict[pd.Timestamp, str]:
    """Map every date in an event window to event labels."""

    lookup: dict[pd.Timestamp, list[str]] = {}
    for row in events.itertuples(index=False):
        start = pd.Timestamp(row.window_start_date).normalize()
        end = pd.Timestamp(row.window_end_date).normalize()
        label = f"{row.name} ({CATEGORY_LABELS.get(row.category, row.category)})"
        for day in pd.date_range(start, end, freq="D"):
            lookup.setdefault(day, []).append(label)
    return {day: "; ".join(names) for day, names in lookup.items()}


def hover_event_text(row: pd.Series) -> str:
    """Prefer exact event name, then event-window context."""

    if row.get("event_name"):
        return str(row["event_name"])
    if row.get("event_window_names"):
        return f"Window: {row['event_window_names']}"
    return "None"


def format_probability(value: object) -> str:
    """Format a probability for hover text."""

    if pd.isna(value):
        return "n/a"
    return f"{float(value):.1%}"


def build_timeline_figure(data: TimelineData) -> go.Figure:
    """Build the interactive Plotly timeline figure."""

    timeline = data.timeline
    activity_metric = data.activity_metric
    activity_label = ACTIVITY_LABELS[activity_metric]
    actual_timeline = timeline[timeline["has_actual_data"]].copy()
    activity_max = max(float(actual_timeline[activity_metric].max()), 1.0)
    figure_width = calculate_timeline_width(timeline)

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.28, 0.72],
        vertical_spacing=0.02,
        specs=[[{}], [{"secondary_y": True}]],
    )
    add_event_windows(figure, data.events, timeline)

    customdata = actual_timeline[
        [
            "oblast_name",
            "alert_count",
            "total_alert_minutes",
            "hover_event_text",
        ]
    ].to_numpy()

    figure.add_trace(
        go.Scatter(
            x=actual_timeline["date"],
            y=actual_timeline[activity_metric],
            mode="lines",
            name=f"Historical {activity_label.lower()}",
            line=dict(color="#1d4ed8", width=2.4),
            customdata=customdata,
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "Oblast: %{customdata[0]}<br>"
                "Actual alert count: %{customdata[1]:,.0f}<br>"
                "Actual alert minutes: %{customdata[2]:,.1f}<br>"
                "Event: %{customdata[3]}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    add_future_forecast_marker(figure, timeline)

    add_event_markers(figure, data.events, timeline)
    style_timeline_figure(
        figure,
        selected_oblast=data.selected_oblast,
        activity_label=activity_label,
        activity_max=activity_max,
        figure_width=figure_width,
    )
    return figure


def calculate_timeline_width(timeline: pd.DataFrame) -> int:
    """Choose a wide fixed canvas so the HTML can scroll horizontally."""

    start = pd.Timestamp(timeline["date"].min())
    end = pd.Timestamp(timeline["date"].max())
    span_days = max(int((end - start).days), 1)
    width = span_days * TIMELINE_PIXELS_PER_DAY
    return min(max(width, TIMELINE_MIN_WIDTH_PX), TIMELINE_MAX_WIDTH_PX)


def add_future_forecast_marker(figure: go.Figure, timeline: pd.DataFrame) -> None:
    """Add forecast only for future dates, never over historical data."""

    future_forecast = timeline[
        timeline["split"].eq("future") & timeline["model_probability"].notna()
    ].copy()
    if future_forecast.empty:
        return

    future_forecast["risk_text"] = future_forecast["model_probability"].map(
        format_probability
    )
    figure.add_trace(
        go.Scatter(
            x=future_forecast["date"],
            y=future_forecast["model_probability"],
            mode="markers+text",
            name="Future next-day risk",
            marker=dict(
                color="#dc2626",
                size=15,
                symbol="circle",
                line=dict(color="white", width=2),
            ),
            text=future_forecast["risk_text"],
            textposition="middle right",
            textfont=dict(color="#dc2626", size=16),
            customdata=future_forecast[
                ["oblast_name", "risk_text", "hover_event_text"]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "Oblast: %{customdata[0]}<br>"
                "Future next-day risk: %{customdata[1]}<br>"
                "Event: %{customdata[2]}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    forecast_date = pd.Timestamp(future_forecast["date"].max())
    figure.add_shape(
        type="line",
        x0=forecast_date,
        x1=forecast_date,
        y0=0,
        y1=1,
        xref="x2",
        yref="paper",
        line=dict(color="#dc2626", width=1.5, dash="dash"),
        layer="above",
    )


def add_event_windows(
    figure: go.Figure,
    events: pd.DataFrame,
    timeline: pd.DataFrame,
) -> None:
    """Add shaded event windows within the visible timeline range."""

    start = timeline["date"].min()
    end = timeline["date"].max()
    for row in events.itertuples(index=False):
        window_start = pd.Timestamp(row.window_start_date)
        window_end = pd.Timestamp(row.window_end_date) + pd.Timedelta(days=1)
        if window_end < start or window_start > end:
            continue
        color = CATEGORY_COLORS.get(row.category, "#64748b")
        figure.add_shape(
            type="rect",
            x0=window_start,
            x1=window_end,
            y0=0,
            y1=1,
            xref="x2",
            yref="paper",
            fillcolor=color,
            opacity=0.07,
            line=dict(width=0),
            layer="below",
        )
        figure.add_shape(
            type="line",
            x0=pd.Timestamp(row.event_date),
            x1=pd.Timestamp(row.event_date),
            y0=0,
            y1=1,
            xref="x2",
            yref="paper",
            line=dict(color=color, width=0.8),
            opacity=0.24,
            layer="below",
        )


def add_event_markers(
    figure: go.Figure,
    events: pd.DataFrame,
    timeline: pd.DataFrame,
) -> None:
    """Add colored event date labels and hoverable markers."""

    start = timeline["date"].min()
    end = timeline["date"].max()
    marker_rows = events[
        (events["event_date"] >= start) & (events["event_date"] <= end)
    ].copy()
    if marker_rows.empty:
        return

    marker_rows = marker_rows.sort_values(
        ["event_date", "category", "name"]
    ).reset_index(drop=True)
    marker_rows["label_lane"] = marker_rows.index % EVENT_LABEL_LANES
    marker_rows["marker_y"] = marker_rows["label_lane"]
    marker_rows["short_label"] = marker_rows.apply(format_event_marker_label, axis=1)

    for category, category_rows in marker_rows.groupby("category", dropna=False):
        color = CATEGORY_COLORS.get(str(category), "#64748b")
        label = CATEGORY_LABELS.get(str(category), str(category))
        figure.add_trace(
            go.Scatter(
                x=category_rows["event_date"],
                y=category_rows["marker_y"],
                mode="markers+text",
                name=f"Event: {label}",
                marker=dict(
                    color=color,
                    size=11,
                    symbol="diamond",
                    line=dict(color="white", width=1),
                ),
                text=category_rows["short_label"],
                textposition="top center",
                textfont=dict(color=color, size=15),
                customdata=category_rows[
                    [
                        "name",
                        "category_label",
                        "window_start_date",
                        "window_end_date",
                        "notes",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Category: %{customdata[1]}<br>"
                    "Window: %{customdata[2]|%Y-%m-%d} to %{customdata[3]|%Y-%m-%d}<br>"
                    "Note: %{customdata[4]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )


def format_event_marker_label(row: pd.Series) -> str:
    """Format event labels with readable names and concrete dates."""

    name = shorten_event_label(str(row["name"]))
    date_label = pd.Timestamp(row["event_date"]).strftime("%d.%m.%Y")
    return f"{name}<br>{date_label}"


def shorten_event_label(name: str) -> str:
    """Create a compact event marker label."""

    replacements = {
        "День пам'яті та перемоги над нацизмом у Другій світовій війні": "8 травня",
        "Річниця повномасштабного вторгнення РФ": "24 лютого",
        "День захисників і захисниць України": "Захисники",
        "День Української Державності": "Державність",
        "День Незалежності України": "Незалежність",
        "Міжнародний жіночий день": "8 березня",
        "День народної єдності РФ": "Єдність РФ",
        "День Конституції України": "Конституція",
    }
    return replacements.get(name, name)


def style_timeline_figure(
    figure: go.Figure,
    *,
    selected_oblast: str,
    activity_label: str,
    activity_max: float,
    figure_width: int,
) -> None:
    """Apply clean presentation styling."""

    figure.update_layout(
        title=dict(
            text=(
                f"Holiday and Symbolic-Date Timeline: {selected_oblast}<br>"
                "<sup>Historical activity only across observed dates; future risk is shown "
                "only at the right edge. Association/event-window comparison, not for "
                "operational use.</sup>"
            ),
            x=0.02,
            xanchor="left",
        ),
        template="plotly_white",
        width=figure_width,
        height=840,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.85)",
            font=dict(size=13),
        ),
        margin=dict(l=82, r=96, t=130, b=90),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Arial, sans-serif", color="#0f172a"),
    )
    figure.update_xaxes(
        row=1,
        col=1,
        showgrid=False,
        showticklabels=False,
        zeroline=False,
    )
    figure.update_xaxes(
        row=2,
        col=1,
        title_text="Date",
        showgrid=True,
        gridcolor="#e2e8f0",
        rangeslider=dict(visible=False),
    )
    figure.update_layout(
        xaxis=dict(domain=[0.0, 0.985]),
        xaxis2=dict(domain=[0.0, 0.985]),
    )
    figure.update_yaxes(
        row=1,
        col=1,
        title_text="",
        range=[-0.6, EVENT_LABEL_LANES + 0.65],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
    )
    figure.update_yaxes(
        row=2,
        col=1,
        title_text=activity_label,
        secondary_y=False,
        showgrid=True,
        gridcolor="#e2e8f0",
        range=[0, activity_max * 1.12],
    )
    figure.update_yaxes(
        row=2,
        col=1,
        title_text="Future next-day risk",
        secondary_y=True,
        tickformat=".0%",
        range=[0, 1],
        showgrid=False,
    )


def save_timeline_html(
    figure: go.Figure,
    output_path: Path = TIMELINE_OUTPUT_PATH,
) -> Path:
    """Save a standalone HTML timeline preview."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_width = int(figure.layout.width or TIMELINE_MIN_WIDTH_PX)
    figure_height = int(figure.layout.height or 840)
    figure_html = pio.to_html(
        figure,
        include_plotlyjs=True,
        full_html=False,
        config={
            "displaylogo": False,
            "responsive": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "holiday_timeline_preview",
                "height": figure_height,
                "width": figure_width,
                "scale": 2,
            },
        },
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(figure.layout.title.text.split('<br>')[0])}</title>
  <style>
    :root {{
      color-scheme: light;
      --page-bg: #f8fafc;
      --frame-border: #cbd5e1;
      --text: #0f172a;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      background: var(--page-bg);
      color: var(--text);
      font-family: Arial, sans-serif;
    }}
    main {{
      padding: 20px;
    }}
    .timeline-frame {{
      width: 100%;
      overflow-x: auto;
      overflow-y: hidden;
      border: 1px solid var(--frame-border);
      border-radius: 8px;
      background: #ffffff;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    }}
    .timeline-inner {{
      min-width: {figure_width}px;
      width: {figure_width}px;
      padding: 6px 8px 0;
    }}
    .timeline-frame::-webkit-scrollbar {{
      height: 14px;
    }}
    .timeline-frame::-webkit-scrollbar-track {{
      background: #e2e8f0;
      border-radius: 999px;
    }}
    .timeline-frame::-webkit-scrollbar-thumb {{
      background: #94a3b8;
      border-radius: 999px;
      border: 3px solid #e2e8f0;
    }}
  </style>
</head>
<body>
  <main>
    <div class="timeline-frame">
      <div class="timeline-inner">
        {figure_html}
      </div>
    </div>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def build_and_save_timeline(
    *,
    oblast: str = DEFAULT_OBLAST,
    activity_metric: str = DEFAULT_ACTIVITY_METRIC,
    output_path: Path = TIMELINE_OUTPUT_PATH,
) -> tuple[go.Figure, TimelineData, Path]:
    """Build and save the one-oblast event timeline."""

    data = prepare_timeline_data(oblast=oblast, activity_metric=activity_metric)
    figure = build_timeline_figure(data)
    saved_path = save_timeline_html(figure, output_path)
    return figure, data, saved_path


def print_timeline_summary(data: TimelineData, output_path: Path) -> None:
    """Print a concise CLI summary."""

    timeline = data.timeline
    visible_events = data.events[
        (data.events["event_date"] >= timeline["date"].min())
        & (data.events["event_date"] <= timeline["date"].max())
    ]
    print("Holiday/event timeline preview generated")
    print("Framing: event-window comparison and exploratory risk only; no causation claim.")
    print(f"Selected oblast: {data.selected_oblast}")
    print(f"Activity metric: {data.activity_metric}")
    print(f"Date range: {timeline['date'].min().date()} to {timeline['date'].max().date()}")
    print(f"Timeline rows: {len(timeline):,}")
    print(f"Visible event markers: {len(visible_events):,}")
    print(f"Saved: {output_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive holiday/event timeline for one oblast."
    )
    parser.add_argument(
        "--oblast",
        default=DEFAULT_OBLAST,
        help='Selected oblast, e.g. "Kyiv Oblast", "Kyiv City", or "Kharkivska oblast".',
    )
    parser.add_argument(
        "--activity-metric",
        choices=sorted(ACTIVITY_LABELS),
        default=DEFAULT_ACTIVITY_METRIC,
        help="Historical activity metric for the left axis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TIMELINE_OUTPUT_PATH,
        help="Output HTML path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _, data, output_path = build_and_save_timeline(
        oblast=args.oblast,
        activity_metric=args.activity_metric,
        output_path=args.output,
    )
    print_timeline_summary(data, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
