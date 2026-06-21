"""Streamlit dashboard for the Ukraine air raid alert analysis project."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from src.timeline_viz import (
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    EVENT_LABEL_LANES,
    shorten_event_label,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DAILY_PATH = PROJECT_ROOT / "data" / "processed" / "daily_oblast_timeseries.csv"
EVENT_LEVEL_PATH = PROJECT_ROOT / "data" / "processed" / "alerts_merged_event_level.csv"
FORECAST_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_daily_oblast.csv"
EVENTS_PATH = PROJECT_ROOT / "data" / "processed" / "expanded_event_calendar.csv"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "oblast_summary.csv"

METRIC_OPTIONS = {
    "alert_count": "Daily alert count",
    "total_alert_minutes": "Daily alert minutes",
    "predicted_risk": "Predicted next-day risk",
}

PLOT_TEMPLATE = "plotly_white"
ALERT_CORAL = "#e4574f"
UKRAINE_BLUE = "#2563eb"
TEXT_DARK = "#172033"


st.set_page_config(
    page_title="Ukraine Air Alert Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --page-bg: #f4efe7;
            --panel-bg: #fffdf8;
            --panel-border: #ded6c9;
            --text-dark: #172033;
            --muted: #6b7280;
            --alert: #e4574f;
            --blue: #2563eb;
        }
        .stApp {
            background: var(--page-bg);
            color: var(--text-dark);
        }
        [data-testid="stSidebar"] {
            background: #eee6da;
            border-right: 1px solid var(--panel-border);
        }
        h1, h2, h3 {
            color: var(--text-dark);
            letter-spacing: 0;
        }
        .project-header {
            padding: 18px 20px;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: linear-gradient(180deg, #fffdf8 0%, #fbf7f0 100%);
            margin-bottom: 16px;
        }
        .project-header h1 {
            margin: 0 0 6px 0;
            font-size: 30px;
            line-height: 1.18;
        }
        .project-header p {
            margin: 0;
            color: var(--muted);
            font-size: 14px;
        }
        .metric-card {
            min-height: 104px;
            padding: 14px 16px;
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: var(--panel-bg);
            box-shadow: 0 1px 2px rgba(23, 32, 51, 0.06);
        }
        .metric-label {
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: .04em;
            margin-bottom: 8px;
        }
        .metric-value {
            color: var(--text-dark);
            font-size: 26px;
            line-height: 1.1;
            font-weight: 700;
        }
        .metric-note {
            margin-top: 6px;
            color: var(--muted);
            font-size: 12px;
        }
        .section-title {
            color: var(--text-dark);
            font-size: 20px;
            font-weight: 700;
            margin: 8px 0 8px 0;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(DAILY_PATH)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["date_kyiv"] = pd.to_datetime(daily["date_kyiv"], errors="coerce")
    for column in ["alert_count", "total_alert_minutes", "average_alert_duration"]:
        daily[column] = pd.to_numeric(daily[column], errors="coerce").fillna(0)

    events = pd.read_csv(EVENT_LEVEL_PATH)
    events["started_at_utc"] = pd.to_datetime(
        events["started_at_utc"], errors="coerce", utc=True
    )
    events["finished_at_utc"] = pd.to_datetime(
        events["finished_at_utc"], errors="coerce", utc=True
    )
    events["duration_minutes"] = pd.to_numeric(
        events["duration_minutes"], errors="coerce"
    )

    forecast = pd.read_csv(FORECAST_PATH)
    forecast["forecast_date"] = pd.to_datetime(
        forecast["forecast_date"], errors="coerce"
    )
    forecast["observation_date"] = pd.to_datetime(
        forecast["observation_date"], errors="coerce"
    )
    forecast["model_probability"] = pd.to_numeric(
        forecast["model_probability"], errors="coerce"
    )

    event_calendar = pd.read_csv(EVENTS_PATH)
    for column in ["event_date", "window_start_date", "window_end_date"]:
        event_calendar[column] = pd.to_datetime(event_calendar[column], errors="coerce")
    event_calendar["category_label"] = event_calendar["category"].map(
        CATEGORY_LABELS
    ).fillna(event_calendar["category"])

    summary = pd.read_csv(SUMMARY_PATH)
    return daily, events, forecast, event_calendar, summary


def format_number(value: float, digits: int = 0) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.{digits}f}"


def format_probability(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_date_range(selected: object, min_date: date, max_date: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    if isinstance(selected, tuple) and len(selected) == 2:
        start, end = selected
    else:
        start, end = min_date, max_date
    return pd.Timestamp(start), pd.Timestamp(end)


def filter_daily(
    daily: pd.DataFrame,
    oblast: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    mask = (
        daily["oblast_name"].eq(oblast)
        & daily["date"].ge(start)
        & daily["date"].le(end)
    )
    return daily.loc[mask].sort_values("date").copy()


def filter_event_level(
    events: pd.DataFrame,
    oblast: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    start_utc = start.tz_localize("UTC")
    end_utc = (end + pd.Timedelta(days=1)).tz_localize("UTC")
    mask = (
        events["oblast_name"].eq(oblast)
        & events["started_at_utc"].ge(start_utc)
        & events["started_at_utc"].lt(end_utc)
    )
    return events.loc[mask].copy()


def latest_future_risk(forecast: pd.DataFrame, oblast: str) -> tuple[float | None, pd.Timestamp | None]:
    rows = forecast[
        forecast["oblast_name"].eq(oblast)
        & forecast["split"].eq("future")
        & forecast["model_probability"].notna()
    ].sort_values("forecast_date")
    if rows.empty:
        return None, None
    latest = rows.iloc[-1]
    return float(latest["model_probability"]), pd.Timestamp(latest["forecast_date"])


def most_active_day(filtered_daily: pd.DataFrame) -> str:
    if filtered_daily.empty or filtered_daily["alert_count"].sum() == 0:
        return "n/a"
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_counts = (
        filtered_daily.assign(day_name=filtered_daily["date"].dt.day_name())
        .groupby("day_name")["alert_count"]
        .sum()
        .reindex(day_order)
    )
    return str(day_counts.idxmax())


def build_regional_table(
    daily: pd.DataFrame,
    forecast: pd.DataFrame,
    summary: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    latest_date = daily["date"].max()
    latest_daily = daily[daily["date"].eq(latest_date)].copy()
    latest_daily = latest_daily[
        ["oblast_name", "alert_count", "total_alert_minutes", "had_alert_binary"]
    ]

    future = forecast[forecast["split"].eq("future")].copy()
    future = future.sort_values("forecast_date").drop_duplicates(
        subset=["oblast_name"], keep="last"
    )
    future = future[["oblast_name", "forecast_date", "model_probability"]]

    table = (
        summary[["oblast_name", "total_alert_hours", "total_alerts"]]
        .merge(latest_daily, on="oblast_name", how="left")
        .merge(future, on="oblast_name", how="left")
    )
    table["latest_alert_minutes"] = table["total_alert_minutes"].fillna(0).round(1)
    table["next_day_risk"] = table["model_probability"].map(format_probability)
    table["total_alert_hours"] = table["total_alert_hours"].round(1)
    sort_column = {
        "alert_count": "alert_count",
        "total_alert_minutes": "latest_alert_minutes",
        "predicted_risk": "model_probability",
    }[metric]
    return table.sort_values(sort_column, ascending=False)[
        [
            "oblast_name",
            "alert_count",
            "latest_alert_minutes",
            "next_day_risk",
            "total_alert_hours",
            "total_alerts",
        ]
    ].rename(
        columns={
            "oblast_name": "Oblast",
            "alert_count": "Latest alert count",
            "latest_alert_minutes": "Latest alert minutes",
            "next_day_risk": "Next-day risk",
            "total_alert_hours": "Historical alert hours",
            "total_alerts": "Historical alerts",
        }
    )


def base_figure_layout(figure: go.Figure, title: str, y_title: str = "") -> go.Figure:
    figure.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=18)),
        template=PLOT_TEMPLATE,
        paper_bgcolor="#fffdf8",
        plot_bgcolor="#fffdf8",
        font=dict(family="Arial, sans-serif", color=TEXT_DARK),
        margin=dict(l=48, r=24, t=58, b=44),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
    )
    figure.update_xaxes(gridcolor="#e8dfd2", linecolor="#d6cbbd")
    figure.update_yaxes(title_text=y_title, gridcolor="#e8dfd2", linecolor="#d6cbbd")
    return figure


def daily_activity_chart(filtered_daily: pd.DataFrame, metric: str) -> go.Figure:
    actual_metric = "alert_count" if metric == "predicted_risk" else metric
    title = f"{METRIC_OPTIONS[actual_metric]} over time"
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=filtered_daily["date"],
            y=filtered_daily[actual_metric],
            mode="lines",
            name=METRIC_OPTIONS[actual_metric],
            line=dict(color=UKRAINE_BLUE, width=2.2),
            customdata=filtered_daily[["alert_count", "total_alert_minutes"]].to_numpy(),
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "Alert count: %{customdata[0]:,.0f}<br>"
                "Alert minutes: %{customdata[1]:,.1f}<extra></extra>"
            ),
        )
    )
    return base_figure_layout(figure, title, METRIC_OPTIONS[actual_metric])


def rolling_average_chart(filtered_daily: pd.DataFrame, metric: str) -> go.Figure:
    actual_metric = "alert_count" if metric == "predicted_risk" else metric
    rolling = filtered_daily[["date", actual_metric]].copy()
    rolling["rolling_7d"] = rolling[actual_metric].rolling(7, min_periods=1).mean()

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=rolling["date"],
            y=rolling["rolling_7d"],
            mode="lines",
            name="7-day rolling average",
            line=dict(color=ALERT_CORAL, width=2.4),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>7-day average: %{y:,.2f}<extra></extra>",
        )
    )
    return base_figure_layout(
        figure,
        f"7-day rolling average: {METRIC_OPTIONS[actual_metric]}",
        "7-day average",
    )


def day_of_week_chart(filtered_daily: pd.DataFrame) -> go.Figure:
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = (
        filtered_daily.assign(day_name=filtered_daily["date"].dt.day_name())
        .groupby("day_name", as_index=False)["alert_count"]
        .sum()
    )
    dow["day_name"] = pd.Categorical(dow["day_name"], day_order, ordered=True)
    dow = dow.sort_values("day_name")

    figure = px.bar(
        dow,
        x="day_name",
        y="alert_count",
        color_discrete_sequence=[UKRAINE_BLUE],
    )
    return base_figure_layout(figure, "Alert count by day of week", "Alert count")


def duration_distribution_chart(events: pd.DataFrame) -> go.Figure:
    finished = events[
        events["duration_minutes"].notna()
        & events["duration_minutes"].ge(0)
        & events["is_finished"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    if finished.empty:
        figure = go.Figure()
        return base_figure_layout(figure, "Alert duration distribution", "Alerts")

    finished["duration_hours"] = finished["duration_minutes"] / 60
    cap = max(float(finished["duration_hours"].quantile(0.99)), 1.0)
    finished["duration_hours_display"] = finished["duration_hours"].clip(upper=cap)
    figure = px.histogram(
        finished,
        x="duration_hours_display",
        nbins=40,
        color_discrete_sequence=[ALERT_CORAL],
    )
    figure.update_traces(
        hovertemplate="Duration: %{x:.2f} hours<br>Alerts: %{y:,}<extra></extra>"
    )
    return base_figure_layout(
        figure,
        "Alert duration distribution",
        "Alerts",
    ).update_xaxes(title_text="Duration hours")


def event_hover_lookup(events: pd.DataFrame) -> tuple[dict[pd.Timestamp, str], dict[pd.Timestamp, str]]:
    exact: dict[pd.Timestamp, list[str]] = {}
    windows: dict[pd.Timestamp, list[str]] = {}
    for row in events.itertuples(index=False):
        event_date = pd.Timestamp(row.event_date).normalize()
        exact.setdefault(event_date, []).append(str(row.name))
        label = f"{row.name} ({CATEGORY_LABELS.get(row.category, row.category)})"
        for day in pd.date_range(row.window_start_date, row.window_end_date, freq="D"):
            windows.setdefault(pd.Timestamp(day).normalize(), []).append(label)
    exact_text = {day: "; ".join(names) for day, names in exact.items()}
    window_text = {day: "; ".join(names) for day, names in windows.items()}
    return exact_text, window_text


def add_event_context(timeline: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    exact, windows = event_hover_lookup(events)
    enriched = timeline.copy()
    normalized_dates = enriched["date"].dt.normalize()
    enriched["event_name"] = normalized_dates.map(exact).fillna("")
    enriched["event_window_names"] = normalized_dates.map(windows).fillna("")
    enriched["hover_event_text"] = enriched["event_name"]
    window_mask = enriched["hover_event_text"].eq("") & enriched["event_window_names"].ne("")
    enriched.loc[window_mask, "hover_event_text"] = (
        "Window: " + enriched.loc[window_mask, "event_window_names"]
    )
    enriched["hover_event_text"] = enriched["hover_event_text"].replace("", "None")
    return enriched


def format_event_marker_label(row: pd.Series) -> str:
    name = shorten_event_label(str(row["name"]))
    date_label = pd.Timestamp(row["event_date"]).strftime("%d.%m.%Y")
    return f"{name}<br>{date_label}"


def build_event_timeline(
    daily: pd.DataFrame,
    forecast: pd.DataFrame,
    event_calendar: pd.DataFrame,
    oblast: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    metric: str,
) -> go.Figure:
    actual_metric = "alert_count" if metric == "predicted_risk" else metric
    filtered_daily = filter_daily(daily, oblast, start, end)
    risk, risk_date = latest_future_risk(forecast, oblast)
    include_future = risk_date is not None and end >= daily["date"].max().normalize()

    timeline = filtered_daily[["date", "oblast_name", "alert_count", "total_alert_minutes"]].copy()
    if include_future and risk_date not in set(timeline["date"]):
        timeline = pd.concat(
            [
                timeline,
                pd.DataFrame(
                    {
                        "date": [risk_date],
                        "oblast_name": [oblast],
                        "alert_count": [None],
                        "total_alert_minutes": [None],
                    }
                ),
            ],
            ignore_index=True,
        )
    timeline["has_actual_data"] = timeline["alert_count"].notna()
    timeline = add_event_context(timeline.sort_values("date"), event_calendar)

    chart_start = timeline["date"].min() if not timeline.empty else start
    chart_end = timeline["date"].max() if not timeline.empty else end
    visible_events = event_calendar[
        event_calendar["event_date"].between(chart_start, chart_end)
    ].copy()
    actual = timeline[timeline["has_actual_data"]].copy()
    activity_max = max(float(actual[actual_metric].max()) if not actual.empty else 1, 1)
    span_days = max(int((pd.Timestamp(chart_end) - pd.Timestamp(chart_start)).days), 1)
    figure_width = min(max(span_days * 6, 1500), 9000)

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.30, 0.70],
        vertical_spacing=0.02,
        specs=[[{}], [{"secondary_y": True}]],
    )

    for row in visible_events.itertuples(index=False):
        color = CATEGORY_COLORS.get(row.category, "#64748b")
        figure.add_shape(
            type="rect",
            x0=row.window_start_date,
            x1=pd.Timestamp(row.window_end_date) + pd.Timedelta(days=1),
            y0=0,
            y1=1,
            xref="x2",
            yref="paper",
            fillcolor=color,
            opacity=0.08,
            line=dict(width=0),
            layer="below",
        )
        figure.add_shape(
            type="line",
            x0=row.event_date,
            x1=row.event_date,
            y0=0,
            y1=1,
            xref="x2",
            yref="paper",
            line=dict(color=color, width=0.8),
            opacity=0.24,
            layer="below",
        )

    figure.add_trace(
        go.Scatter(
            x=actual["date"],
            y=actual[actual_metric],
            mode="lines",
            name=METRIC_OPTIONS[actual_metric],
            line=dict(color=UKRAINE_BLUE, width=2.4),
            customdata=actual[["alert_count", "total_alert_minutes", "hover_event_text"]].to_numpy(),
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "Actual alert count: %{customdata[0]:,.0f}<br>"
                "Actual alert minutes: %{customdata[1]:,.1f}<br>"
                "Event: %{customdata[2]}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    if include_future and risk is not None and risk_date is not None:
        figure.add_trace(
            go.Scatter(
                x=[risk_date],
                y=[risk],
                mode="markers+text",
                name="Future next-day risk",
                marker=dict(
                    color=ALERT_CORAL,
                    size=15,
                    symbol="circle",
                    line=dict(color="#fffdf8", width=2),
                ),
                text=[format_probability(risk)],
                textposition="middle right",
                textfont=dict(color=ALERT_CORAL, size=16),
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d}</b><br>"
                    f"Oblast: {oblast}<br>"
                    "Future next-day risk: %{text}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
            secondary_y=True,
        )
        figure.add_shape(
            type="line",
            x0=risk_date,
            x1=risk_date,
            y0=0,
            y1=1,
            xref="x2",
            yref="paper",
            line=dict(color=ALERT_CORAL, width=1.5, dash="dash"),
            layer="above",
        )

    if not visible_events.empty:
        marker_rows = visible_events.sort_values(
            ["event_date", "category", "name"]
        ).reset_index(drop=True)
        marker_rows["marker_y"] = marker_rows.index % EVENT_LABEL_LANES
        marker_rows["label"] = marker_rows.apply(format_event_marker_label, axis=1)
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
                        size=10,
                        symbol="diamond",
                        line=dict(color="#fffdf8", width=1),
                    ),
                    text=category_rows["label"],
                    textposition="top center",
                    textfont=dict(color=color, size=14),
                    customdata=category_rows[
                        ["name", "category_label", "window_start_date", "window_end_date"]
                    ].to_numpy(),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Category: %{customdata[1]}<br>"
                        "Window: %{customdata[2]|%Y-%m-%d} to "
                        "%{customdata[3]|%Y-%m-%d}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

    figure.update_layout(
        title=dict(
            text=(
                f"Holiday and symbolic-date timeline: {oblast}<br>"
                "<sup>Actual historical activity with future risk shown only at the right edge.</sup>"
            ),
            x=0.02,
            xanchor="left",
            font=dict(size=18),
        ),
        width=figure_width,
        height=760,
        template=PLOT_TEMPLATE,
        paper_bgcolor="#fffdf8",
        plot_bgcolor="#fffdf8",
        font=dict(family="Arial, sans-serif", color=TEXT_DARK),
        margin=dict(l=72, r=90, t=110, b=66),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.01, x=0, font=dict(size=12)),
        xaxis=dict(domain=[0.0, 0.985]),
        xaxis2=dict(domain=[0.0, 0.985]),
    )
    figure.update_xaxes(row=1, col=1, showticklabels=False, showgrid=False)
    figure.update_xaxes(row=2, col=1, title_text="Date", gridcolor="#e8dfd2")
    figure.update_yaxes(
        row=1,
        col=1,
        range=[-0.6, EVENT_LABEL_LANES + 0.65],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )
    figure.update_yaxes(
        row=2,
        col=1,
        title_text=METRIC_OPTIONS[actual_metric],
        secondary_y=False,
        range=[0, activity_max * 1.12],
        gridcolor="#e8dfd2",
    )
    figure.update_yaxes(
        row=2,
        col=1,
        title_text="Future next-day risk",
        secondary_y=True,
        range=[0, 1],
        tickformat=".0%",
        showgrid=False,
    )
    return figure


def render_scrollable_plotly(figure: go.Figure, height: int = 810) -> None:
    figure_width = int(figure.layout.width or 1600)
    html = pio.to_html(
        figure,
        include_plotlyjs=True,
        full_html=False,
        config={
            "displaylogo": False,
            "responsive": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "event_timeline",
                "height": int(figure.layout.height or 760),
                "width": figure_width,
                "scale": 2,
            },
        },
    )
    components.html(
        f"""
        <div style="width:100%; overflow-x:auto; overflow-y:hidden; background:#fffdf8;
                    border:1px solid #ded6c9; border-radius:8px; padding:6px;">
          <div style="width:{figure_width}px; min-width:{figure_width}px;">
            {html}
          </div>
        </div>
        """,
        height=height,
        scrolling=False,
    )


def main() -> None:
    inject_css()
    daily, events, forecast, event_calendar, summary = load_data()

    min_date = daily["date"].min().date()
    max_date = daily["date"].max().date()
    oblasts = sorted(daily["oblast_name"].dropna().unique())
    default_oblast = "Kyivska oblast" if "Kyivska oblast" in oblasts else oblasts[0]

    with st.sidebar:
        st.header("Filters")
        selected_oblast = st.selectbox(
            "Oblast",
            oblasts,
            index=oblasts.index(default_oblast),
        )
        selected_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        metric = st.selectbox(
            "Metric",
            options=list(METRIC_OPTIONS.keys()),
            format_func=METRIC_OPTIONS.get,
        )

    start_date, end_date = normalize_date_range(selected_range, min_date, max_date)
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    filtered_daily = filter_daily(daily, selected_oblast, start_date, end_date)
    filtered_events = filter_event_level(events, selected_oblast, start_date, end_date)
    risk, risk_date = latest_future_risk(forecast, selected_oblast)
    latest_data_date = daily["date"].max()

    st.markdown(
        f"""
        <div class="project-header">
          <h1>Time Series Analysis of Air Raid Alerts in Ukraine</h1>
          <p>Exploratory analysis and event-window comparison only; not for operational prediction. Data freshness: {latest_data_date:%Y-%m-%d}.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered_daily.empty:
        st.warning("No daily records found for this oblast and date range.")
        return

    total_alerts = float(filtered_daily["alert_count"].sum())
    total_minutes = float(filtered_daily["total_alert_minutes"].sum())
    avg_duration = total_minutes / total_alerts if total_alerts else 0

    card_cols = st.columns(6)
    with card_cols[0]:
        metric_card("Total alerts", format_number(total_alerts), "Selected range")
    with card_cols[1]:
        metric_card("Alert hours", format_number(total_minutes / 60, 1), "Selected range")
    with card_cols[2]:
        metric_card("Average duration", f"{avg_duration:.1f} min", "Weighted by alerts")
    with card_cols[3]:
        metric_card("Most active day", most_active_day(filtered_daily), "By alert count")
    with card_cols[4]:
        metric_card("Latest data date", f"{latest_data_date:%Y-%m-%d}", "Daily data")
    with card_cols[5]:
        metric_card(
            "Next-day risk",
            format_probability(risk),
            f"Forecast date {risk_date:%Y-%m-%d}" if risk_date is not None else "Unavailable",
        )

    st.markdown('<div class="section-title">Regional overview</div>', unsafe_allow_html=True)
    regional_table = build_regional_table(daily, forecast, summary, metric)
    st.dataframe(regional_table, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Alert activity</div>', unsafe_allow_html=True)
    top_left, top_right = st.columns(2)
    with top_left:
        st.plotly_chart(
            daily_activity_chart(filtered_daily, metric),
            use_container_width=True,
            theme=None,
        )
    with top_right:
        st.plotly_chart(
            rolling_average_chart(filtered_daily, metric),
            use_container_width=True,
            theme=None,
        )

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.plotly_chart(day_of_week_chart(filtered_daily), use_container_width=True, theme=None)
    with bottom_right:
        st.plotly_chart(
            duration_distribution_chart(filtered_events),
            use_container_width=True,
            theme=None,
        )

    st.markdown('<div class="section-title">Holiday and symbolic-date timeline</div>', unsafe_allow_html=True)
    timeline_figure = build_event_timeline(
        daily,
        forecast,
        event_calendar,
        selected_oblast,
        start_date,
        end_date,
        metric,
    )
    render_scrollable_plotly(timeline_figure)


if __name__ == "__main__":
    main()
