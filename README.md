# Time Series Analysis of Air Raid Alerts in Ukraine

Interactive analytical dashboard for exploring Ukrainian air raid alert history, symbolic-date event windows, and simple 60-day forecast scenarios by oblast.

> **Disclaimer:** this project is for exploratory data analysis and educational comparison only. It is **not for operational use**, emergency decision-making, real-time safety guidance, or military prediction.

## Jury Quick Start

The fastest way to assess the project is to run the Streamlit dashboard:

```powershell
python -m streamlit run app.py
```

Recommended dashboard walkthrough:

1. Use the map at the top to select an oblast.
2. Open **Overview** to inspect high-level alert totals and regional comparison.
3. Open **Historical** to review daily activity, rolling averages, day-of-week patterns, and duration distribution.
4. Open **Events timeline** to inspect the holiday/symbolic-date timeline.
5. Open **Forecast**, enable **Future mode**, and review the 60-day forecast section.

The dashboard is intentionally framed as analytical. It compares alert activity around dates and forecast scenarios; it does not claim causation or operational predictive reliability.

## What The Project Does

The project combines long-term historical alert data with a recent API delta layer, normalizes both sources into one event-level schema, aggregates alerts into oblast-level time series, and presents the result as an interactive dashboard.

Core implemented functionality:

- Interactive Ukraine oblast map selector using Plotly and Streamlit selection events.
- Historical alert exploration by oblast.
- Daily and hourly time series preprocessing.
- Summary statistics by oblast.
- Seasonality statistics by day of week, month, and hour where available.
- Static analysis figures saved under `outputs/figures/`.
- Holiday and symbolic-date event calendar.
- Horizontal event timeline with shaded event windows and readable date labels.
- Next-day alert-risk baseline models.
- 60-day forecast output for probability, expected alert count, and expected alert minutes.
- Dedicated **View Into The Future** dashboard mode with dark blue forecast styling.

Current local processed dataset snapshot:

- Event-level merged alerts: about `158,051` rows.
- Covered oblasts: `25`.
- Historical daily range: `2022-03-15` to `2026-06-21`.
- 60-day forecast range: `2026-06-22` to `2026-08-20`.

## Key Dashboard Features

### Interactive Map

The map is the primary region selector. Clicking an oblast updates the selected region for all lower dashboard views. The map can show:

- historical alert count;
- historical alert minutes;
- predicted next-day risk in forecast/future mode.

Crimea and Sevastopol are visible in the boundary data but are shown as no-data regions if they are not present in the processed alert dataset.

### Historical Analysis

The historical section lets the assessor inspect:

- daily alert count;
- total alert minutes;
- optional rolling average;
- day-of-week pattern;
- duration distribution;
- filtered event-level records.

The goal is to make it easy to compare regions and identify broad temporal patterns without hiding the underlying data.

### Holiday And Symbolic-Date Functionality

One of the main project ideas is to compare air raid alert activity around Ukrainian and Russian public, military, memorial, and symbolic dates.

The event calendar is stored in:

```text
data/events/holiday_events.csv
```

It is expanded into concrete yearly dates and event windows:

```text
data/processed/expanded_event_calendar.csv
```

Implemented event categories:

- `ukrainian_public`
- `ukrainian_memorial`
- `russian_state`
- `russian_military`
- `shared_public`
- `other_symbolic`

Examples of included dates:

- New Year;
- Christmas;
- Ukrainian Constitution Day;
- Ukrainian Statehood Day;
- Ukrainian Independence Day;
- Day of Defenders of Ukraine;
- 8 May remembrance date;
- 9 May Russian Victory Day;
- Russian Airborne Forces Day;
- Russian Unity Day;
- Orthodox Easter as a moving holiday.

The purpose is not to prove that a holiday causes alerts. The project uses neutral analytical language: **association**, **comparison**, **event window**, and **uplift**. I tried to explore whether alert activity around these symbolic dates looks different from ordinary periods. The dashboard supports visual comparison through shaded windows, event markers, hover labels, and the future-mode event horizon card.

### View Into The Future

The Forecast tab includes a **Future mode** switch. When enabled, it turns the forecast section into a dark blue "View Into The Future" surface.

It shows:

- next-day alert probability;
- predicted alerts over the next 60 days;
- predicted alert minutes/hours over the next 60 days;
- highest-risk forecast date;
- active symbolic dates in the forecast horizon;
- future timeline after the latest observed data date;
- visual divider between historical observations and future predictions.

Forecast values are exploratory expected values, not operational predictions.

## Forecasting Status

Implemented forecast outputs are saved to:

```text
data/processed/forecast_daily_oblast.csv
data/processed/forecast_60d_oblast.csv
outputs/tables/forecast_metrics.json
outputs/tables/forecast_60d_metrics.json
```

Implemented baseline models:

- `baseline_rolling_7d`
- `baseline_same_day_of_week`
- `baseline_event_adjusted`

Optional/experimental model path:

- `lstm_multitask_v1`

The LSTM implementation is present as an optional PyTorch path. It is not the main validated model. It was planned as the stronger forecasting extension, but full tuning, calibration, and robust jury-ready validation were not completed before submission. The app and pipeline therefore remain usable with baseline models even if PyTorch is unavailable.

## Repository Structure

```text
app.py                         Streamlit dashboard
main.py                        CLI entry point
src/api_client.py              alerts.in.ua API client
src/fetch_github_dataset.py    historical GitHub dataset ingestion
src/fetch_api_delta.py         recent API delta ingestion
src/normalize.py               source schema normalization
src/merge_sources.py           GitHub/API source merge and deduplication
src/preprocess_timeseries.py   daily/hourly oblast time series preprocessing
src/analyze.py                 statistical analysis and static figures
src/events.py                  holiday/symbolic-date calendar expansion
src/forecast.py                next-day and 60-day forecast generation
src/timeline_viz.py            standalone Plotly event timeline
data/events/holiday_events.csv source symbolic-date calendar
data/geo/                      Ukraine oblast GeoJSON
scripts/api_smoke_test.py      API connection smoke test
docs/                          API notes
```

## Setup

Create and activate a Python environment, then install dependencies:

```powershell
pip install -r requirements.txt
```

Optional LSTM dependencies:

```powershell
pip install -r requirements-ml.txt
```

Create `.env` with the alerts.in.ua API token:

```text
ALERTS_IN_UA_TOKEN=your_token_here
```

The token is needed for API exploration and recent delta ingestion. The long-term historical baseline comes from the public GitHub dataset.

## Running The Pipeline

Typical execution order:

```powershell
python src\fetch_github_dataset.py
python src\fetch_api_delta.py --regions all
python src\merge_sources.py
python src\preprocess_timeseries.py
python main.py events
python main.py analyze --oblast "Kyivska oblast"
python main.py forecast --horizon 60 --models baselines
python main.py timeline --oblast "Kyivska oblast"
python -m streamlit run app.py
```

Optional LSTM run:

```powershell
python main.py forecast --horizon 60 --models baselines,lstm
```

If PyTorch is missing or incompatible, the baseline forecast path should still work.

## Data Sources

Main historical source:

- `Vadimkin/ukrainian-air-raid-sirens-dataset`

Recent delta / API exploration source:

- `alerts.in.ua`

Map boundaries:

- geoBoundaries Ukraine ADM1 GeoJSON.

The project preserves raw source files where possible and keeps processed files separate from raw data.

## Outputs To Inspect

Important processed data:

```text
data/processed/alerts_merged_event_level.csv
data/processed/daily_oblast_timeseries.csv
data/processed/hourly_oblast_timeseries.csv
data/processed/expanded_event_calendar.csv
data/processed/forecast_60d_oblast.csv
```

Important tables:

```text
outputs/tables/oblast_summary.csv
outputs/tables/seasonality_summary.csv
outputs/tables/forecast_metrics.json
outputs/tables/forecast_60d_metrics.json
```

Important visual outputs:

```text
outputs/figures/
outputs/holiday_timeline_preview.html
```

## What Was Planned But Not Fully Completed

The following items were planned but were not completed to full submission quality:

- Full LSTM model tuning and calibration.
- Deep model comparison dashboard with per-oblast validation charts.
- Forecast uncertainty intervals.
- More robust event-uplift statistical testing.
- Export/download controls for filtered dashboard data.
- More polished mobile layout for every dashboard view.
- A fully custom JavaScript map; the current implementation uses Plotly + Streamlit native selection instead.

These are intentionally documented as limitations, not hidden as finished features.

## Limitations

- The forecast is exploratory and should not be used operationally.
- Event-window comparisons do not prove causation.
- Some data sources can lag, change schema, or be incomplete.
- Forecast quality depends on historical alert reporting consistency.
- The LSTM path is optional and not required for the dashboard to run.
- Processed data files may be large and are ignored by git in some cases; regenerate them with the pipeline if missing.

## Assessment Notes

The project is strongest as an end-to-end applied data product:

- real data ingestion;
- source normalization;
- event-level to time-series preprocessing;
- statistical analysis;
- symbolic-date comparison;
- baseline forecasting;
- interactive map-based dashboard;
- clear separation between historical observations and future estimates.

The central idea to assess is the **holiday/symbolic-date timeline**: it gives a visual way to compare alert activity around selected Ukrainian and Russian dates while explicitly avoiding causal claims.
