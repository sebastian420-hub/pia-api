import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List

import asyncpg
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config  # loads .env first
from auth import ws_token_ok
from routers import router as api_router
from sensors_router import router as sensors_router, live_session_reaper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pia-api")


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                self.disconnect(connection)


manager = ConnectionManager()


async def listen_to_pg_notify():
    """Dedicated connection that LISTENs for new_intelligence and fans out to WebSockets."""
    while True:
        conn = None
        try:
            conn = await asyncpg.connect(config.DATABASE_URL)
            logger.info("Connected to PostgreSQL for LISTEN/NOTIFY.")

            def handle_notify(connection, pid, channel, payload):
                # asyncpg invokes this on the event loop thread
                asyncio.get_running_loop().create_task(manager.broadcast(payload))

            await conn.add_listener('new_intelligence', handle_notify)
            while not conn.is_closed():
                await asyncio.sleep(5)
            logger.warning("LISTEN connection closed; reconnecting.")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"PostgreSQL LISTEN error: {e}; retrying in 5s")
            await asyncio.sleep(5)
        finally:
            if conn is not None and not conn.is_closed():
                await conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await asyncpg.create_pool(config.DATABASE_URL, min_size=1, max_size=10)
    app.state.pg_task = asyncio.create_task(listen_to_pg_notify())
    app.state.reaper_task = asyncio.create_task(live_session_reaper(app.state.pool))
    if not config.PIA_API_TOKEN:
        logger.error("PIA_API_TOKEN is not set: every authenticated route will answer 503.")
    yield
    app.state.pg_task.cancel()
    app.state.reaper_task.cancel()
    await app.state.pool.close()


app = FastAPI(lifespan=lifespan, title="PIA API Bridge", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")
app.include_router(sensors_router, prefix="/api/v1")


# The UI checks `status === 'success'` and shows `message` otherwise, so error
# bodies keep that shape while carrying a real HTTP status code.
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"status": "error", "message": exc.detail},
                        headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        content={"status": "error", "message": "Invalid request", "errors": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        content={"status": "error", "message": "Internal server error"})


@app.get("/")
def read_root():
    return {"status": "online", "message": "PIA Bridge API is running."}


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """Live intelligence feed. Requires ?token=<PIA_API_TOKEN> on the handshake."""
    if not ws_token_ok(websocket):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
