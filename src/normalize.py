"""Normalize alert source datasets into a shared event-level schema."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from fetch_github_dataset import OFFICIAL_DATASET_URL, RAW_DATASET_PATH
except ImportError:
    from .fetch_github_dataset import OFFICIAL_DATASET_URL, RAW_DATASET_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_DELTA_PATH = PROJECT_ROOT / "data" / "processed" / "api_delta_alerts.csv"

GITHUB_DATA_SOURCE = "github_official"
API_DATA_SOURCE = "alerts_api"

UNIFIED_COLUMNS = [
    "source_record_id",
    "data_source",
    "region_name_original",
    "oblast_name",
    "raion_name",
    "location_type",
    "alert_type",
    "started_at_utc",
    "finished_at_utc",
    "duration_minutes",
    "is_finished",
    "source_url",
    "source_note",
]

GITHUB_REQUIRED_COLUMNS = {
    "oblast",
    "raion",
    "hromada",
    "level",
    "started_at",
    "finished_at",
    "source",
}

API_REQUIRED_COLUMNS = {
    "data_source",
    "source_id",
    "source_url",
    "region_name",
    "region_type",
    "oblast_name",
    "raion_name",
    "alert_type",
    "started_at",
    "finished_at",
    "duration_minutes",
}

UKRAINIAN_TO_GITHUB_OBLAST_NAME = {
    "Автономна Республіка Крим": "Crimea",
    "АР Крим": "Crimea",
    "Вінницька область": "Vinnytska oblast",
    "Волинська область": "Volynska oblast",
    "Дніпропетровська область": "Dnipropetrovska oblast",
    "Донецька область": "Donetska oblast",
    "Житомирська область": "Zhytomyrska oblast",
    "Закарпатська область": "Zakarpatska oblast",
    "Запорізька область": "Zaporizka oblast",
    "Івано-Франківська область": "Ivano-Frankivska oblast",
    "Київська область": "Kyivska oblast",
    "Кіровоградська область": "Kirovohradska oblast",
    "Луганська область": "Luhanska oblast",
    "Львівська область": "Lvivska oblast",
    "Миколаївська область": "Mykolaivska oblast",
    "Одеська область": "Odeska oblast",
    "Полтавська область": "Poltavska oblast",
    "Рівненська область": "Rivnenska oblast",
    "Сумська область": "Sumska oblast",
    "Тернопільська область": "Ternopilska oblast",
    "Харківська область": "Kharkivska oblast",
    "Херсонська область": "Khersonska oblast",
    "Хмельницька область": "Khmelnytska oblast",
    "Черкаська область": "Cherkaska oblast",
    "Чернівецька область": "Chernivetska oblast",
    "Чернігівська область": "Chernihivska oblast",
    "м. Київ": "Kyiv City",
    "Київ": "Kyiv City",
    "м. Севастополь": "Sevastopol City",
    "Севастополь": "Sevastopol City",
}


def normalize_github_alerts(raw_path: Path = RAW_DATASET_PATH) -> pd.DataFrame:
    """Normalize the official GitHub historical CSV."""

    if not raw_path.exists():
        raise FileNotFoundError(
            f"GitHub raw dataset not found: {raw_path}. "
            "Run src/fetch_github_dataset.py first."
        )

    dataframe = pd.read_csv(raw_path, dtype="string")
    require_columns(dataframe.columns, GITHUB_REQUIRED_COLUMNS, source="GitHub")

    normalized = pd.DataFrame(index=dataframe.index)
    normalized["data_source"] = GITHUB_DATA_SOURCE
    normalized["region_name_original"] = dataframe.apply(
        most_specific_github_location,
        axis=1,
    )
    normalized["oblast_name"] = dataframe["oblast"].map(standardize_oblast_name)
    normalized["raion_name"] = clean_string_series(dataframe["raion"])
    normalized["location_type"] = clean_string_series(dataframe["level"])
    normalized["alert_type"] = "air_raid"
    normalized["started_at_utc"] = to_utc(dataframe["started_at"])
    normalized["finished_at_utc"] = to_utc(dataframe["finished_at"])
    normalized["duration_minutes"] = calculate_duration_minutes(
        normalized["started_at_utc"],
        normalized["finished_at_utc"],
    )
    normalized["is_finished"] = normalized["finished_at_utc"].notna()
    normalized["source_url"] = OFFICIAL_DATASET_URL
    normalized["source_note"] = clean_string_series(dataframe["source"])
    normalized.insert(
        0,
        "source_record_id",
        build_record_ids(
            GITHUB_DATA_SOURCE,
            dataframe[
                [
                    "oblast",
                    "raion",
                    "hromada",
                    "level",
                    "started_at",
                    "finished_at",
                    "source",
                ]
            ],
        ),
    )

    return normalized[UNIFIED_COLUMNS]


def normalize_api_delta(api_delta_path: Path = API_DELTA_PATH) -> pd.DataFrame:
    """Normalize the alerts.in.ua API delta CSV."""

    if not api_delta_path.exists():
        raise FileNotFoundError(
            f"API delta dataset not found: {api_delta_path}. "
            "Run src/fetch_api_delta.py first."
        )

    dataframe = pd.read_csv(api_delta_path, dtype="string")
    require_columns(dataframe.columns, API_REQUIRED_COLUMNS, source="API delta")

    normalized = pd.DataFrame(index=dataframe.index)
    normalized["data_source"] = API_DATA_SOURCE
    normalized["region_name_original"] = clean_string_series(dataframe["region_name"])
    normalized["oblast_name"] = dataframe["oblast_name"].map(standardize_oblast_name)
    normalized["raion_name"] = clean_string_series(dataframe["raion_name"])
    normalized["location_type"] = clean_string_series(dataframe["region_type"])
    normalized["alert_type"] = clean_string_series(dataframe["alert_type"])
    normalized["started_at_utc"] = to_utc(dataframe["started_at"])
    normalized["finished_at_utc"] = to_utc(dataframe["finished_at"])

    api_duration = pd.to_numeric(dataframe["duration_minutes"], errors="coerce")
    calculated_duration = calculate_duration_minutes(
        normalized["started_at_utc"],
        normalized["finished_at_utc"],
    )
    normalized["duration_minutes"] = api_duration.fillna(calculated_duration)
    normalized["is_finished"] = normalized["finished_at_utc"].notna()
    normalized["source_url"] = clean_string_series(dataframe["source_url"])
    normalized["source_note"] = pd.NA
    normalized.insert(
        0,
        "source_record_id",
        build_api_record_ids(dataframe),
    )

    return normalized[UNIFIED_COLUMNS]


def to_utc(series: pd.Series) -> pd.Series:
    """Parse timestamps as timezone-aware UTC values."""

    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")


def calculate_duration_minutes(
    started_at: pd.Series,
    finished_at: pd.Series,
) -> pd.Series:
    """Calculate event duration in minutes where both timestamps are present."""

    duration = (finished_at - started_at).dt.total_seconds() / 60
    return duration.where(finished_at.notna(), pd.NA)


def standardize_oblast_name(value: object) -> object:
    """Map API Ukrainian oblast names to the GitHub English oblast style."""

    if pd.isna(value):
        return pd.NA

    cleaned = str(value).strip()
    if not cleaned:
        return pd.NA

    return UKRAINIAN_TO_GITHUB_OBLAST_NAME.get(cleaned, cleaned)


def most_specific_github_location(row: pd.Series) -> object:
    """Return hromada, then raion, then oblast from a GitHub source row."""

    for column in ("hromada", "raion", "oblast"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return pd.NA


def build_record_ids(data_source: str, values: pd.DataFrame) -> list[str]:
    """Build deterministic record IDs for sources without stable IDs."""

    return [
        f"{data_source}:{hash_record(row)}"
        for row in values.astype("string").fillna("").itertuples(index=False, name=None)
    ]


def build_api_record_ids(dataframe: pd.DataFrame) -> list[str]:
    """Use the API source ID when available, otherwise fall back to a hash."""

    ids: list[str] = []
    fallback_columns = [
        "region_name",
        "region_type",
        "oblast_name",
        "alert_type",
        "started_at",
        "finished_at",
    ]
    fallback_ids = build_record_ids(API_DATA_SOURCE, dataframe[fallback_columns])

    for source_id, fallback_id in zip(dataframe["source_id"], fallback_ids):
        if pd.notna(source_id) and str(source_id).strip():
            ids.append(f"{API_DATA_SOURCE}:{str(source_id).strip()}")
        else:
            ids.append(fallback_id)

    return ids


def hash_record(values: Iterable[object]) -> str:
    """Hash row values into a compact deterministic identifier."""

    text = "|".join("" if pd.isna(value) else str(value) for value in values)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def clean_string_series(series: pd.Series) -> pd.Series:
    """Strip strings and convert empty values to pandas NA."""

    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned == "")


def require_columns(columns: Iterable[str], required: set[str], *, source: str) -> None:
    """Raise a clear error if an input source schema changed."""

    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"{source} dataset missing required columns: {missing}")
