"""Small alerts.in.ua API client for data-source exploration.

This module intentionally covers only the API connection layer. It does not
clean, analyze, forecast, or visualize alert data.

Documented API coverage:
- Active alerts: GET /v1/alerts/active.json
- Compact active air-raid statuses by oblast:
  GET /v1/iot/active_air_raid_alerts_by_oblast.json
- Compact active air-raid statuses by UID:
  GET /v1/iot/active_air_raid_alerts/{uid}.json
- Compact active air-raid statuses for all UIDs:
  GET /v1/iot/active_air_raid_alerts.json
- Region history: GET /v1/regions/{uid}/alerts/{period}.json

The public docs list oblast and special-city UIDs directly. They also link a
Google Sheet with the wider rayon/hromada UID list, but no dedicated locations
API endpoint is documented.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import find_dotenv, load_dotenv


API_BASE_URL = "https://api.alerts.in.ua"
TOKEN_ENV_VAR = "ALERTS_IN_UA_TOKEN"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MIN_INTERVAL_SECONDS = 7.5
HISTORY_PERIOD_MONTH_AGO = "month_ago"

# The docs state a soft limit of 8-10 requests/minute and a hard limit of
# 12 requests/minute for API calls from one IP address. The history endpoint has
# its own stricter limit of 2 requests/minute.
SOFT_LIMIT_REQUESTS_PER_MINUTE = "8-10"
HARD_LIMIT_REQUESTS_PER_MINUTE = 12
HISTORY_LIMIT_REQUESTS_PER_MINUTE = 2


OBLAST_AND_SPECIAL_CITY_LOCATIONS: tuple[dict[str, str], ...] = (
    {"uid": "3", "name": "Хмельницька область", "type": "oblast"},
    {"uid": "4", "name": "Вінницька область", "type": "oblast"},
    {"uid": "5", "name": "Рівненська область", "type": "oblast"},
    {"uid": "8", "name": "Волинська область", "type": "oblast"},
    {"uid": "9", "name": "Дніпропетровська область", "type": "oblast"},
    {"uid": "10", "name": "Житомирська область", "type": "oblast"},
    {"uid": "11", "name": "Закарпатська область", "type": "oblast"},
    {"uid": "12", "name": "Запорізька область", "type": "oblast"},
    {"uid": "13", "name": "Івано-Франківська область", "type": "oblast"},
    {"uid": "14", "name": "Київська область", "type": "oblast"},
    {"uid": "15", "name": "Кіровоградська область", "type": "oblast"},
    {"uid": "16", "name": "Луганська область", "type": "oblast"},
    {"uid": "17", "name": "Миколаївська область", "type": "oblast"},
    {"uid": "18", "name": "Одеська область", "type": "oblast"},
    {"uid": "19", "name": "Полтавська область", "type": "oblast"},
    {"uid": "20", "name": "Сумська область", "type": "oblast"},
    {"uid": "21", "name": "Тернопільська область", "type": "oblast"},
    {"uid": "22", "name": "Харківська область", "type": "oblast"},
    {"uid": "23", "name": "Херсонська область", "type": "oblast"},
    {"uid": "24", "name": "Черкаська область", "type": "oblast"},
    {"uid": "25", "name": "Чернігівська область", "type": "oblast"},
    {"uid": "26", "name": "Чернівецька область", "type": "oblast"},
    {"uid": "27", "name": "Львівська область", "type": "oblast"},
    {"uid": "28", "name": "Донецька область", "type": "oblast"},
    {"uid": "29", "name": "Автономна Республіка Крим", "type": "oblast"},
    {"uid": "30", "name": "м. Севастополь", "type": "special_city"},
    {"uid": "31", "name": "м. Київ", "type": "special_city"},
)


class AlertsInUaError(RuntimeError):
    """Base exception for alerts.in.ua client errors."""


class MissingTokenError(AlertsInUaError):
    """Raised when the API token is not configured."""


class AlertsInUaHTTPError(AlertsInUaError):
    """Raised for non-success HTTP responses."""


class AlertsInUaRateLimitError(AlertsInUaHTTPError):
    """Raised when the API reports rate limiting."""


class AlertsInUaResponseError(AlertsInUaError):
    """Raised when a response does not match the expected schema."""


def load_environment(
    dotenv_path: str | os.PathLike[str] | None = None,
    *,
    override: bool = False,
) -> bool:
    """Load environment variables from a .env file using python-dotenv."""

    if dotenv_path is None:
        dotenv_file = find_dotenv(usecwd=True)
    else:
        dotenv_file = str(Path(dotenv_path))

    if not dotenv_file:
        return False

    return load_dotenv(dotenv_path=dotenv_file, override=override)


@dataclass(frozen=True)
class ResponseMeta:
    """Minimal response metadata useful during API exploration."""

    status_code: int
    url: str
    content_type: str | None
    last_modified: str | None
    retry_after: str | None


@dataclass(frozen=True)
class APIResult:
    """Decoded API data plus selected HTTP metadata."""

    data: Any
    meta: ResponseMeta


class AlertsInUaClient:
    """Synchronous client for the documented alerts.in.ua endpoints."""

    def __init__(
        self,
        token: str | None = None,
        *,
        base_url: str = API_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        if token is None:
            load_environment()

        self.token = token or os.getenv(TOKEN_ENV_VAR)
        if not self.token:
            raise MissingTokenError(
                f"Missing API token. Set {TOKEN_ENV_VAR} in the environment "
                "or in a local .env file."
            )

        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self.session = session or requests.Session()
        self._last_request_at: float | None = None

    def list_documented_locations(self) -> list[dict[str, str]]:
        """Return the oblast/special-city UID list published in the docs."""

        return [location.copy() for location in OBLAST_AND_SPECIAL_CITY_LOCATIONS]

    def get_active_alerts(self) -> APIResult:
        """Fetch currently active alerts."""

        result = self._get_json("/v1/alerts/active.json")
        self._validate_alerts_payload(result.data, endpoint="active alerts")
        return result

    def get_alert_history(
        self, uid: int | str, period: str = HISTORY_PERIOD_MONTH_AGO
    ) -> APIResult:
        """Fetch documented alert history for one region and period."""

        if period != HISTORY_PERIOD_MONTH_AGO:
            raise ValueError(
                "Unsupported history period. The public docs currently list only "
                f"{HISTORY_PERIOD_MONTH_AGO!r}."
            )

        safe_uid = self._normalize_uid(uid)
        result = self._get_json(f"/v1/regions/{safe_uid}/alerts/{period}.json")
        self._validate_alerts_payload(result.data, endpoint="alert history")
        return result

    def get_air_raid_status_by_uid(self, uid: int | str) -> APIResult:
        """Fetch compact active air-raid status for a documented UID."""

        safe_uid = self._normalize_uid(uid)
        result = self._get_json(
            f"/v1/iot/active_air_raid_alerts/{safe_uid}.json",
            allow_plain_text=True,
        )
        self._validate_status_string(result.data, endpoint="status by UID")
        return result

    def get_air_raid_statuses_by_oblast(self) -> APIResult:
        """Fetch compact active air-raid statuses in the documented oblast order."""

        result = self._get_json(
            "/v1/iot/active_air_raid_alerts_by_oblast.json",
            allow_plain_text=True,
        )
        self._validate_status_string(result.data, endpoint="statuses by oblast")
        return result

    def get_all_air_raid_statuses(self) -> APIResult:
        """Fetch compact active air-raid statuses for all UID indexes."""

        result = self._get_json(
            "/v1/iot/active_air_raid_alerts.json",
            allow_plain_text=True,
        )
        self._validate_status_string(
            result.data,
            endpoint="all UID statuses",
            allow_spaces=True,
        )
        return result

    def _get_json(self, path: str, *, allow_plain_text: bool = False) -> APIResult:
        response = self._request("GET", path)
        data = self._decode_response(response, allow_plain_text=allow_plain_text)
        return APIResult(data=data, meta=self._response_meta(response))

    def _request(self, method: str, path: str) -> requests.Response:
        self._respect_rate_limit()

        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "kse-alerts-timeseries-api-exploration/0.1",
        }

        try:
            response = self.session.request(
                method,
                url,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except requests.Timeout as exc:
            raise AlertsInUaError(
                f"Request to alerts.in.ua timed out after {self.timeout_seconds}s: {url}"
            ) from exc
        except requests.RequestException as exc:
            raise AlertsInUaError(f"Request to alerts.in.ua failed: {exc}") from exc

        self._last_request_at = time.monotonic()
        self._raise_for_status(response)
        return response

    def _respect_rate_limit(self) -> None:
        if self._last_request_at is None or self.min_interval_seconds <= 0:
            return

        elapsed = time.monotonic() - self._last_request_at
        remaining = self.min_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _raise_for_status(self, response: requests.Response) -> None:
        if response.status_code in {200, 304}:
            return

        message = self._extract_error_message(response)
        endpoint = response.url.split("?", 1)[0]

        if response.status_code == 401:
            raise AlertsInUaHTTPError(
                f"alerts.in.ua returned 401 Unauthorized for {endpoint}. "
                f"Check {TOKEN_ENV_VAR}. Message: {message}"
            )
        if response.status_code == 403:
            raise AlertsInUaHTTPError(
                f"alerts.in.ua returned 403 Forbidden for {endpoint}. "
                f"Your IP may be blocked or the API may be unavailable in this country. "
                f"Message: {message}"
            )
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            retry_hint = f" Retry-After: {retry_after}s." if retry_after else ""
            raise AlertsInUaRateLimitError(
                f"alerts.in.ua returned 429 Too Many Requests for {endpoint}."
                f"{retry_hint} Documented soft limit: {SOFT_LIMIT_REQUESTS_PER_MINUTE} "
                f"requests/minute; hard limit: {HARD_LIMIT_REQUESTS_PER_MINUTE}/minute; "
                f"history endpoint limit: {HISTORY_LIMIT_REQUESTS_PER_MINUTE}/minute."
            )

        raise AlertsInUaHTTPError(
            f"alerts.in.ua returned HTTP {response.status_code} for {endpoint}. "
            f"Message: {message}"
        )

    def _decode_response(
        self, response: requests.Response, *, allow_plain_text: bool
    ) -> Any:
        if response.status_code == 304:
            return None

        text = response.text.strip("\ufeff\r\n")
        if not text:
            raise AlertsInUaResponseError(
                f"Empty response body from {response.url.split('?', 1)[0]}"
            )

        try:
            return response.json()
        except ValueError as exc:
            if allow_plain_text and self._looks_like_status_string(text):
                return text
            raise AlertsInUaResponseError(
                f"Expected JSON from {response.url.split('?', 1)[0]}, "
                f"got Content-Type {response.headers.get('Content-Type')!r}."
            ) from exc

    def _extract_error_message(self, response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            message = payload.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()

        text = response.text.strip()
        return text[:300] if text else "<no response message>"

    def _response_meta(self, response: requests.Response) -> ResponseMeta:
        return ResponseMeta(
            status_code=response.status_code,
            url=response.url.split("?", 1)[0],
            content_type=response.headers.get("Content-Type"),
            last_modified=response.headers.get("Last-Modified"),
            retry_after=response.headers.get("Retry-After"),
        )

    def _validate_alerts_payload(self, data: Any, *, endpoint: str) -> None:
        if not isinstance(data, dict):
            raise AlertsInUaResponseError(
                f"Expected {endpoint} response to be an object, got {type(data).__name__}."
            )

        alerts = data.get("alerts")
        if not isinstance(alerts, list):
            raise AlertsInUaResponseError(
                f"Expected {endpoint} response to include an 'alerts' list."
            )

        for index, alert in enumerate(alerts[:5]):
            if not isinstance(alert, dict):
                raise AlertsInUaResponseError(
                    f"Expected alert item {index} in {endpoint} response to be an object."
                )

    def _validate_status_string(
        self,
        data: Any,
        *,
        endpoint: str,
        allow_spaces: bool = False,
    ) -> None:
        if not isinstance(data, str):
            raise AlertsInUaResponseError(
                f"Expected {endpoint} response to be a string, got {type(data).__name__}."
            )

        allowed = {"N", "A", "P"}
        if allow_spaces:
            allowed.add(" ")

        unexpected = sorted(set(data) - allowed)
        if unexpected:
            raise AlertsInUaResponseError(
                f"Unexpected symbols in {endpoint} response: {unexpected!r}."
            )

    def _looks_like_status_string(self, text: str) -> bool:
        return bool(text) and set(text) <= {"N", "A", "P", " "}

    def _normalize_uid(self, uid: int | str) -> str:
        uid_text = str(uid).strip()
        if not uid_text.isdigit():
            raise ValueError(f"UID must be numeric, got {uid!r}.")
        return uid_text
