"""Dash review UI for the historical alerts pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dash_table, dcc, html


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DAILY_PATH = PROCESSED_DIR / "daily_oblast_timeseries.csv"
EVENTS_PATH = PROCESSED_DIR / "alerts_merged_event_level.csv"
MERGE_REPORT_PATH = PROCESSED_DIR / "source_merge_report.json"
PREPROCESS_REPORT_PATH = PROCESSED_DIR / "timeseries_preprocess_report.json"

SOURCE_LABELS = {
    "github_official": "GitHub official",
    "alerts_api": "alerts.in.ua API",
}
METRIC_LABELS = {
    "alert_count": "Alert count",
    "total_alert_minutes": "Total alert minutes",
    "average_alert_duration": "Average duration",
}
COLOR_SEQUENCE = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4f46e5",
]


def load_daily_timeseries(path: Path = DAILY_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(path)
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    dataframe["date_kyiv"] = pd.to_datetime(dataframe["date_kyiv"], errors="coerce")
    return dataframe.dropna(subset=["date", "oblast_name"])


def load_event_level_alerts(path: Path = EVENTS_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(path)
    dataframe["started_at_utc"] = pd.to_datetime(
        dataframe["started_at_utc"],
        errors="coerce",
        utc=True,
        format="mixed",
    )
    dataframe["finished_at_utc"] = pd.to_datetime(
        dataframe["finished_at_utc"],
        errors="coerce",
        utc=True,
        format="mixed",
    )
    dataframe["started_date"] = dataframe["started_at_utc"].dt.tz_localize(None).dt.date
    dataframe["duration_minutes"] = pd.to_numeric(
        dataframe["duration_minutes"],
        errors="coerce",
    )
    return dataframe.dropna(subset=["started_at_utc", "oblast_name"])


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


daily_df = load_daily_timeseries()
events_df = load_event_level_alerts()
merge_report = load_json(MERGE_REPORT_PATH)
preprocess_report = load_json(PREPROCESS_REPORT_PATH)


app = Dash(__name__, title="Air Raid Alerts Review")
server = app.server


def build_layout() -> html.Div:
    if daily_df.empty or events_df.empty:
        return html.Div(
            className="app-shell",
            children=[
                html.Main(
                    className="content-only",
                    children=[
                        html.H1("Air Raid Alerts Review"),
                        html.P(
                            "Processed data files are missing or empty. Run the "
                            "ingestion, merge, and preprocessing scripts first.",
                            className="notice",
                        ),
                    ],
                )
            ],
        )

    oblasts = sorted(daily_df["oblast_name"].dropna().unique().tolist())
    default_oblasts = ["Kyiv City"] if "Kyiv City" in oblasts else oblasts[:1]
    sources = sorted(events_df["data_source"].dropna().unique().tolist())
    location_types = sorted(events_df["location_type"].dropna().unique().tolist())

    return html.Div(
        className="app-shell",
        children=[
            html.Aside(
                className="sidebar",
                children=[
                    html.Div(
                        className="brand",
                        children=[
                            html.H1("Air Raid Alerts Review"),
                            html.P(
                                "Educational historical data review only. Not for "
                                "operational, safety, emergency-response, or military use.",
                                className="disclaimer",
                            ),
                        ],
                    ),
                    filter_block(
                        "Oblast",
                        dcc.Dropdown(
                            id="oblast-filter",
                            options=[{"label": value, "value": value} for value in oblasts],
                            value=default_oblasts,
                            multi=True,
                            clearable=False,
                        ),
                    ),
                    filter_block(
                        "Date range",
                        dcc.DatePickerRange(
                            id="date-filter",
                            min_date_allowed=daily_df["date"].min().date(),
                            max_date_allowed=daily_df["date"].max().date(),
                            start_date=daily_df["date"].min().date(),
                            end_date=daily_df["date"].max().date(),
                            display_format="YYYY-MM-DD",
                        ),
                    ),
                    filter_block(
                        "Source",
                        dcc.Dropdown(
                            id="source-filter",
                            options=[
                                {"label": SOURCE_LABELS.get(value, value), "value": value}
                                for value in sources
                            ],
                            value=sources,
                            multi=True,
                            clearable=False,
                        ),
                    ),
                    filter_block(
                        "Location type",
                        dcc.Dropdown(
                            id="location-filter",
                            options=[
                                {"label": value, "value": value}
                                for value in location_types
                            ],
                            value=location_types,
                            multi=True,
                            clearable=False,
                        ),
                    ),
                    filter_block(
                        "Daily metric",
                        dcc.RadioItems(
                            id="metric-filter",
                            options=[
                                {"label": label, "value": value}
                                for value, label in METRIC_LABELS.items()
                            ],
                            value="alert_count",
                            className="radio-stack",
                        ),
                    ),
                ],
            ),
            html.Main(
                className="main",
                children=[
                    html.Section(id="metric-row", className="metric-row"),
                    dcc.Tabs(
                        id="review-tabs",
                        value="daily",
                        className="tabs",
                        children=[
                            dcc.Tab(label="Daily history", value="daily"),
                            dcc.Tab(label="Source coverage", value="sources"),
                            dcc.Tab(label="Event records", value="events"),
                            dcc.Tab(label="Data quality", value="quality"),
                        ],
                    ),
                    html.Section(id="tab-content", className="tab-content"),
                ],
            ),
        ],
    )


def filter_block(label: str, child: Any) -> html.Div:
    return html.Div(
        className="filter-block",
        children=[html.Label(label), child],
    )


app.layout = build_layout


@app.callback(
    Output("metric-row", "children"),
    Output("tab-content", "children"),
    Input("oblast-filter", "value"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
    Input("source-filter", "value"),
    Input("location-filter", "value"),
    Input("metric-filter", "value"),
    Input("review-tabs", "value"),
)
def update_dashboard(
    oblasts: list[str] | None,
    start_date: str | None,
    end_date: str | None,
    sources: list[str] | None,
    location_types: list[str] | None,
    metric: str,
    active_tab: str,
) -> tuple[list[html.Div], html.Section]:
    filters = normalize_filters(
        oblasts=oblasts,
        start_date=start_date,
        end_date=end_date,
        sources=sources,
        location_types=location_types,
        metric=metric,
    )
    daily = filter_daily(daily_df, filters)
    events = filter_events(events_df, filters)

    metrics = render_metrics(daily, events)
    if active_tab == "daily":
        content = render_daily_tab(daily, metric)
    elif active_tab == "sources":
        content = render_source_tab(daily, events)
    elif active_tab == "events":
        content = render_events_tab(events)
    else:
        content = render_quality_tab()

    return metrics, content


def normalize_filters(
    *,
    oblasts: list[str] | None,
    start_date: str | None,
    end_date: str | None,
    sources: list[str] | None,
    location_types: list[str] | None,
    metric: str,
) -> dict[str, Any]:
    return {
        "oblasts": oblasts or [],
        "start_date": pd.Timestamp(start_date or daily_df["date"].min()),
        "end_date": pd.Timestamp(end_date or daily_df["date"].max()),
        "sources": sources or [],
        "location_types": location_types or [],
        "metric": metric,
    }


def filter_daily(dataframe: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    mask = (
        dataframe["oblast_name"].isin(filters["oblasts"])
        & (dataframe["date"] >= filters["start_date"])
        & (dataframe["date"] <= filters["end_date"])
    )
    return dataframe.loc[mask].copy()


def filter_events(dataframe: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    start_date = filters["start_date"].date()
    end_date = filters["end_date"].date()
    mask = (
        dataframe["oblast_name"].isin(filters["oblasts"])
        & dataframe["data_source"].isin(filters["sources"])
        & dataframe["location_type"].isin(filters["location_types"])
        & (dataframe["started_date"] >= start_date)
        & (dataframe["started_date"] <= end_date)
    )
    return dataframe.loc[mask].copy()


def render_metrics(daily: pd.DataFrame, events: pd.DataFrame) -> list[html.Div]:
    total_alerts = int(daily["alert_count"].sum()) if not daily.empty else 0
    total_hours = (
        float(daily["total_alert_minutes"].sum()) / 60 if not daily.empty else 0.0
    )
    active_days = int(daily.loc[daily["had_alert_binary"] == 1, "date"].nunique())

    return [
        metric_card("Daily alert starts", f"{total_alerts:,}"),
        metric_card("Alert-hours", f"{total_hours:,.1f}"),
        metric_card("Active days", f"{active_days:,}"),
        metric_card("Event rows", f"{len(events):,}"),
    ]


def metric_card(label: str, value: str) -> html.Div:
    return html.Div(
        className="metric-card",
        children=[html.Span(label), html.Strong(value)],
    )


def render_daily_tab(daily: pd.DataFrame, metric: str) -> html.Section:
    if daily.empty:
        return empty_state("No daily rows match the selected filters.")

    metric_label = METRIC_LABELS[metric]
    trend = px.line(
        daily,
        x="date",
        y=metric,
        color="oblast_name",
        labels={"date": "Date", metric: metric_label, "oblast_name": "Oblast"},
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    style_figure(trend, height=430)

    monthly = daily.copy()
    monthly["month"] = monthly["date"].dt.to_period("M").astype("string")
    monthly = (
        monthly.groupby(["month", "oblast_name"], dropna=False)
        .agg(total_alert_minutes=("total_alert_minutes", "sum"))
        .reset_index()
    )
    monthly_chart = px.bar(
        monthly,
        x="month",
        y="total_alert_minutes",
        color="oblast_name",
        labels={
            "month": "Month",
            "total_alert_minutes": "Total alert minutes",
            "oblast_name": "Oblast",
        },
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    style_figure(monthly_chart, height=360)

    return html.Section(
        children=[
            dcc.Graph(figure=trend, config={"displayModeBar": False}),
            dcc.Graph(figure=monthly_chart, config={"displayModeBar": False}),
        ]
    )


def render_source_tab(daily: pd.DataFrame, events: pd.DataFrame) -> html.Section:
    if daily.empty or events.empty:
        return empty_state("No source coverage rows match the selected filters.")

    source_daily = daily.melt(
        id_vars=["date", "oblast_name"],
        value_vars=["github_records_count", "api_records_count"],
        var_name="source",
        value_name="records",
    )
    source_daily["source"] = source_daily["source"].map(
        {
            "github_records_count": "GitHub official",
            "api_records_count": "alerts.in.ua API",
        }
    )
    source_daily = source_daily[source_daily["records"] > 0]

    daily_chart = px.bar(
        source_daily,
        x="date",
        y="records",
        color="source",
        labels={"date": "Date", "records": "Records", "source": "Source"},
        color_discrete_sequence=["#2563eb", "#dc2626"],
    )
    style_figure(daily_chart, height=380)

    source_counts = (
        events["data_source"]
        .map(lambda value: SOURCE_LABELS.get(value, value))
        .value_counts()
        .reset_index()
    )
    source_counts.columns = ["source", "records"]
    pie = px.pie(
        source_counts,
        names="source",
        values="records",
        hole=0.45,
        color_discrete_sequence=["#2563eb", "#dc2626"],
    )
    style_figure(pie, height=380)

    return html.Section(
        className="split-grid",
        children=[
            dcc.Graph(figure=daily_chart, config={"displayModeBar": False}),
            dcc.Graph(figure=pie, config={"displayModeBar": False}),
        ],
    )


def render_events_tab(events: pd.DataFrame) -> html.Section:
    if events.empty:
        return empty_state("No event records match the selected filters.")

    display_columns = [
        "started_at_utc",
        "finished_at_utc",
        "oblast_name",
        "region_name_original",
        "raion_name",
        "location_type",
        "alert_type",
        "duration_minutes",
        "data_source",
        "source_record_id",
    ]
    table = events[display_columns].copy()
    table["data_source"] = table["data_source"].map(
        lambda value: SOURCE_LABELS.get(value, value)
    )
    table["started_at_utc"] = table["started_at_utc"].astype("string")
    table["finished_at_utc"] = table["finished_at_utc"].astype("string")
    table["duration_minutes"] = table["duration_minutes"].round(2)
    table = table.sort_values("started_at_utc", ascending=False).head(1000)

    return html.Section(
        children=[
            dash_table.DataTable(
                data=table.to_dict("records"),
                columns=[{"name": column, "id": column} for column in table.columns],
                page_size=25,
                sort_action="native",
                filter_action="native",
                fixed_rows={"headers": True},
                style_table={"height": "620px", "overflowY": "auto"},
                style_header={
                    "backgroundColor": "#f1f5f9",
                    "fontWeight": "700",
                    "border": "1px solid #dbe3ef",
                },
                style_cell={
                    "fontFamily": "Inter, Segoe UI, sans-serif",
                    "fontSize": "13px",
                    "padding": "8px",
                    "border": "1px solid #e2e8f0",
                    "textAlign": "left",
                    "maxWidth": "260px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                },
            )
        ]
    )


def render_quality_tab() -> html.Section:
    return html.Section(
        className="split-grid",
        children=[
            html.Div(
                className="json-panel",
                children=[
                    html.H3("Merge"),
                    html.Pre(json.dumps(merge_report, ensure_ascii=False, indent=2)),
                ],
            ),
            html.Div(
                className="json-panel",
                children=[
                    html.H3("Preprocessing"),
                    html.Pre(json.dumps(preprocess_report, ensure_ascii=False, indent=2)),
                ],
            ),
        ],
    )


def empty_state(message: str) -> html.Section:
    return html.Section(className="empty-state", children=message)


def style_figure(figure: Any, *, height: int) -> None:
    figure.update_layout(
        height=height,
        margin=dict(l=24, r=24, t=28, b=24),
        hovermode="x unified",
        legend_title_text="",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#0f172a"),
    )
    figure.update_xaxes(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1")
    figure.update_yaxes(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1")


if __name__ == "__main__":
    app.run(
        host=os.getenv("DASH_HOST", "127.0.0.1"),
        port=int(os.getenv("DASH_PORT", "8050")),
        debug=False,
    )
