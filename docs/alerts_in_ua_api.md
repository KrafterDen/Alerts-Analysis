# alerts.in.ua API Exploration Notes

This project uses alerts.in.ua data for educational time-series analysis only.
It is not for operational, safety, emergency-response, or military use.

Source documentation: https://devs.alerts.in.ua/

## Authentication

Set the token in a local `.env` file:

```text
ALERTS_IN_UA_TOKEN=your-token
```

The file is ignored by Git and should not be committed.

Alternatively, set the token in the shell environment:

```powershell
$env:ALERTS_IN_UA_TOKEN = "your-token"
```

The client sends it as:

```text
Authorization: Bearer <token>
```

## Endpoint Coverage

| Need | Documented endpoint / source | Notes |
| --- | --- | --- |
| List regions / locations | No dedicated API endpoint documented | Docs provide oblast and special-city UIDs. Full rayon/hromada UID list is linked as a Google Sheet. |
| Active alerts | `GET /v1/alerts/active.json` | Returns an object with an `alerts` list. |
| Active air-raid statuses by oblast | `GET /v1/iot/active_air_raid_alerts_by_oblast.json` | Returns a compact status string in documented oblast order. |
| Active air-raid status by UID | `GET /v1/iot/active_air_raid_alerts/{uid}.json` | Returns one compact status character. |
| Active air-raid statuses for all UIDs | `GET /v1/iot/active_air_raid_alerts.json` | Returns a compact status string where position corresponds to UID. |
| Alert history by region and period | `GET /v1/regions/{uid}/alerts/{period}.json` | Docs currently list `month_ago` as the available period. This endpoint has a separate 2 requests/minute limit. |

## Alert Response Schema

`/v1/alerts/active.json` and `/v1/regions/{uid}/alerts/month_ago.json`
return an object with an `alerts` list. The active-alert response observed in
the smoke test also included `meta` and `disclaimer` top-level fields.

```json
{
  "alerts": [
    {
      "id": 10,
      "location_title": "Луганська область",
      "location_type": "oblast",
      "started_at": "2022-04-04T16:45:39.000Z",
      "finished_at": null,
      "updated_at": "2022-04-08T08:04:26.316Z",
      "alert_type": "air_raid",
      "location_uid": "16",
      "location_oblast": "Луганська область",
      "location_oblast_uid": "16",
      "location_raion": "Луганський район",
      "country": null,
      "deleted_at": null,
      "notes": "За повідомленям голови ОВА",
      "calculated": false
    }
  ],
  "meta": {
    "last_updated_at": "2026-06-21T09:20:08.000Z",
    "type": "full"
  },
  "disclaimer": "..."
}
```

Documented `location_type` values:

```text
oblast, raion, city, hromada, unknown
```

Documented `alert_type` values:

```text
air_raid, artillery_shelling, urban_fights, chemical, nuclear
```

## Compact Status Schema

The IoT endpoints return compact strings:

```text
A = active air raid alert
P = partial alert in raions or hromadas
N = no air-raid alert information
space = no data for UID, only documented for the all-UID endpoint
```

## Rate Limits

The docs list:

- Soft limit: 8-10 requests per minute from one IP address.
- Hard limit: 12 requests per minute from one IP address.
- History endpoint limit: 2 requests per minute.
- `429 Too Many Requests` means the client should slow down.

The local client adds a default pause between sequential calls and raises a
specific rate-limit error on HTTP 429.
