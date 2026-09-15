"""
PIA US relay — a deliberately tiny fetch proxy.

Deploy on any US host (Fly.io machine, $5 VPS). It fetches from an allow-list of
upstream hosts only, requires a bearer token, caps size and time, and does nothing else.

    RELAY_TOKEN=...  uvicorn relay:app --host 0.0.0.0 --port 8080
"""
import os
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response

TOKEN = os.getenv("RELAY_TOKEN", "")
ALLOWED_HOSTS = tuple(h.strip() for h in os.getenv(
    "RELAY_ALLOWED_HOSTS",
    "webcams.nyctmc.org,cwwp2.dot.ca.gov,cwwp2.dot.ca.gov,wzcam.dot.ca.gov,cctv.dot.ca.gov"
).split(",") if h.strip())
MAX_BYTES = int(os.getenv("RELAY_MAX_BYTES", str(4 * 1024 * 1024)))

app = FastAPI(title="PIA relay", docs_url=None, redoc_url=None)
client = httpx.AsyncClient(timeout=20, follow_redirects=True, headers={"User-Agent": "PIA-relay/1.0"})


def _allowed(host: str) -> bool:
    return any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/fetch")
async def fetch(request: Request, url: str = Query(..., max_length=2048)):
    auth = request.headers.get("authorization", "")
    if not TOKEN or auth != f"Bearer {TOKEN}":
        raise HTTPException(401, "bad token")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not _allowed(parts.hostname or ""):
        raise HTTPException(403, "host not allowed")
    async with client.stream("GET", url) as resp:
        if resp.status_code != 200:
            raise HTTPException(502, f"upstream {resp.status_code}")
        ctype = resp.headers.get("content-type", "application/octet-stream")
        chunks, total = [], 0
        async for chunk in resp.aiter_bytes():
            total += len(chunk)
            if total > MAX_BYTES:
                raise HTTPException(502, "upstream too large")
            chunks.append(chunk)
    body = b"".join(chunks)
    # HLS playlists reference segments by absolute URL; rewrite them to pass through the relay too.
    if ctype.startswith("application/vnd.apple.mpegurl") or url.endswith(".m3u8"):
        base = f"{request.url.scheme}://{request.url.netloc}/fetch?url="
        lines = []
        for line in body.decode("utf-8", "replace").splitlines():
            if line and not line.startswith("#") and line.startswith("http"):
                line = base + line
            lines.append(line)
        body = "\n".join(lines).encode()
    return Response(body, media_type=ctype, headers={"Cache-Control": "max-age=2"})
