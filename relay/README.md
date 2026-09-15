# PIA US relay

New York and Caltrans camera feeds only answer requests from US IP addresses. This is a
one-file proxy you run on a US host; PIA calls it instead of the upstream sites.

- Allow-listed hosts only (`RELAY_ALLOWED_HOSTS`), bearer token (`RELAY_TOKEN`), 4 MB cap, 20 s timeout.
- HLS playlists are rewritten so video segments also pass through the relay.
- `GET /healthz` → `{"ok": true}`; `GET /fetch?url=…` with `Authorization: Bearer <token>`.

Cheapest way to run it on demand: Fly.io (`fly.toml.example`) — the machine stops when idle and
PIA's live-session start/stop hooks turn it on and off. A plain $5 VPS with `docker run` works too
(then leave `RELAY_START_CMD`/`RELAY_STOP_CMD` empty and the relay is treated as always on).

Local test:
```
RELAY_TOKEN=x uvicorn relay:app --port 8080
curl -H "Authorization: Bearer x" "localhost:8080/fetch?url=https://webcams.nyctmc.org/api/cameras"
```
