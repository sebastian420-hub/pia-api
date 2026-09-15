"""
Bearer-token authentication.

Every REST route depends on `require_token`; the WebSocket calls `ws_token_ok`
on the handshake. A missing PIA_API_TOKEN fails closed (503) so a misconfigured
deployment can never be open by accident.
"""
import secrets

from fastapi import Depends, HTTPException, Request, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import PIA_API_TOKEN

_bearer = HTTPBearer(auto_error=False)

# <img>/<video> tags cannot send headers; these GET paths may carry ?token= instead.
_QUERY_TOKEN_SUFFIXES = ("/snapshot", "/video")


def _token_matches(candidate: str) -> bool:
    return bool(candidate) and secrets.compare_digest(candidate, PIA_API_TOKEN)


def require_token(request: Request, creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> None:
    if not PIA_API_TOKEN:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "PIA_API_TOKEN is not configured on the server")
    if creds is not None and creds.scheme.lower() == "bearer" and _token_matches(creds.credentials):
        return
    if request.method == "GET" and request.url.path.endswith(_QUERY_TOKEN_SUFFIXES) \
            and _token_matches(request.query_params.get("token", "")):
        return
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid bearer token",
                        headers={"WWW-Authenticate": "Bearer"})


def ws_token_ok(websocket: WebSocket) -> bool:
    """Accepts ?token=... or an Authorization: Bearer header on the WebSocket handshake."""
    if not PIA_API_TOKEN:
        return False
    token = websocket.query_params.get("token", "")
    if not token:
        header = websocket.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            token = header[7:].strip()
    return _token_matches(token)
