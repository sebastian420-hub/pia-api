import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import List

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
from routers import router as api_router
import shutil

# Load environment variables
load_dotenv()

# Setup Document Directory for Uploads
DOC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pia-core", "data", "documents"))
os.makedirs(DOC_DIR, exist_ok=True)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "pia")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "pia")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# LLM Setup
llm_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o")

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
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")

manager = ConnectionManager()

async def listen_to_pg_notify(app: FastAPI):
    """Connects to PostgreSQL and listens for intelligence notifications."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        logger.info("Connected to PostgreSQL for LISTEN/NOTIFY.")

        def handle_notify(connection, pid, channel, payload):
            logger.info(f"Received notification on channel {channel}: {payload}")
            # Broadcast the payload to all connected WebSocket clients
            asyncio.create_task(manager.broadcast(payload))

        await conn.add_listener('new_intelligence', handle_notify)
        
        # Keep connection alive
        while True:
            await asyncio.sleep(3600)
    except Exception as e:
        logger.error(f"PostgreSQL connection error: {e}")
    finally:
        if 'conn' in locals() and not conn.is_closed():
            await conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create a background task for PostgreSQL LISTEN
    app.state.pg_task = asyncio.create_task(listen_to_pg_notify(app))
    yield
    # Shutdown: Cancel the background task
    app.state.pg_task.cancel()

app = FastAPI(lifespan=lifespan, title="PIA API Bridge", version="1.0.0")

app.include_router(api_router, prefix="/api/v1")

# Allow the frontend (CesiumJS/React) to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []

@app.post("/api/v1/chat")
async def chat_endpoint(request: ChatRequest):
    """AI Co-Pilot endpoint for tactical interrogation of the Knowledge Graph."""
    try:
        # We perform a basic semantic search across recent UIRs to provide context
        conn = await asyncpg.connect(DATABASE_URL)
        # 1. Look for entities matching keywords in the prompt to provide context
        query = """
            SELECT content_headline, content_summary, entities, priority 
            FROM intelligence_records 
            WHERE created_at > NOW() - INTERVAL '7 days'
            ORDER BY created_at DESC 
            LIMIT 10;
        """
        recent_intel = await conn.fetch(query)
        await conn.close()
        
        intel_context = "\n".join([f"- [{r['priority']}] {r['content_headline']}: {r['content_summary']}" for r in recent_intel])

        system_prompt = f"""
        You are the Tactical AI Co-Pilot of the Personal Intelligence Agency (PIA).
        The user (Director) is interrogating you via the Live Dashboard.
        
        Provide concise, tactical, military-grade intelligence summaries based strictly on the provided context.
        Do not hallucinate facts. If the answer is not in the context, state that data is unavailable.
        Keep responses under 3 paragraphs. Use bullet points for readability.
        
        CURRENT RECENT INTELLIGENCE CONTEXT:
        {intel_context}
        """

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(request.history)
        messages.append({"role": "user", "content": request.message})

        response = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2
        )
        
        return {"status": "success", "reply": response.choices[0].message.content}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/")
def read_root():
    return {"status": "online", "message": "PIA Bridge API is running."}

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for the React frontend to subscribe to live events."""
    await manager.connect(websocket)
    try:
        while True:
            # We don't expect the client to send much, but we need to keep connection open
            data = await websocket.receive_text()
            logger.debug(f"Received from client: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/v1/clusters/active")
async def get_active_clusters():
    """Fetches currently active intelligence clusters for the map."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        # Assuming table name is 'intelligence_clusters' or similar from Layer 3
        # We handle this gracefully if table isn't populated yet
        query = """
            SELECT 
                cluster_id, 
                title as name, 
                status, 
                confidence,
                priority,
                domain,
                ST_Y(geo_centroid) as lat,
                ST_X(geo_centroid) as lon
            FROM intelligence_clusters 
            WHERE status = 'ACTIVE' 
            LIMIT 50;
        """
        try:
            records = await conn.fetch(query)
            return {"status": "success", "data": [dict(r) for r in records]}
        except asyncpg.exceptions.UndefinedTableError:
            return {"status": "success", "data": [], "message": "No clusters table yet or no active clusters."}
        finally:
            await conn.close()
    except Exception as e:
        logger.error(f"Error fetching clusters: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/v1/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Receives a document and saves it for the document_agent to process."""
    try:
        file_location = os.path.join(DOC_DIR, file.filename)
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        logger.info(f"Received document upload: {file.filename}")
        return {"status": "success", "message": f"File '{file.filename}' securely uploaded to the ingestion queue."}
    except Exception as e:
        logger.error(f"Failed to upload document: {e}")
        return {"status": "error", "message": f"Failed to upload: {str(e)}"}

@app.get("/api/v1/event/{uid}")
async def get_event_details(uid: str):
    """Fetches the detailed AI SITREP and extracted entities for a specific intelligence record."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        query = """
            SELECT content_summary, entities 
            FROM intelligence_records 
            WHERE uid = $1 
            LIMIT 1;
        """
        record = await conn.fetchrow(query, uid)
        await conn.close()
        
        if not record:
            return {"status": "error", "message": "Record not found"}
            
        return {
            "status": "success", 
            "data": {
                "summary": record['content_summary'] or "No AI summary available.",
                "entities": record['entities'] or []
            }
        }
    except Exception as e:
        logger.error(f"Error fetching event details: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/v1/logs")
async def get_system_logs():
    """Fetches the latest agent activity across all tables to display in the UI Terminal."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        # Union the analysis queue (Analyst Agent) with intelligence records (Ingestor Agents)
        query = """
            SELECT created_at, agent, action, message, status
            FROM (
                -- Analyst Agent Logs
                SELECT 
                    created_at, 
                    COALESCE(assigned_agent, 'SYSTEM') as agent, 
                    trigger_type as action, 
                    COALESCE(error_message, 'Job ID: ' || queue_id::text) as message,
                    status
                FROM analysis_queue 
                
                UNION ALL
                
                -- Ingestor Agent Logs (News, Seismic, Aviation, Maritime, Document)
                SELECT 
                    created_at, 
                    source_agent as agent, 
                    'INGEST_' || source_type as action, 
                    content_headline as message,
                    'DONE' as status
                FROM intelligence_records
            ) combined_logs
            ORDER BY created_at DESC 
            LIMIT 30;
        """
        records = await conn.fetch(query)
        await conn.close()
        
        logs = []
        for r in records:
            time_str = r['created_at'].strftime("%H:%M:%S")
            agent = r['agent']
            status = r['status']
            action = r['action']
            message = r['message']
            
            if status == 'FAILED':
                msg = f"[{time_str}] [{agent}] ERROR: {message}"
            elif status == 'PROCESSING':
                msg = f"[{time_str}] [{agent}] PROCESSING: {action}"
            else:
                msg = f"[{time_str}] [{agent}] {action}: {message}"
            logs.append(msg)
            
        return {"status": "success", "data": logs}
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/v1/archive")
async def get_intelligence_archive(page: int = 1, limit: int = 50):
    """Fetches paginated historical intelligence records for the Archive Dashboard."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        offset = (page - 1) * limit
        query = """
            SELECT uid, created_at, source_type, priority, domain, content_headline, content_summary, entities
            FROM intelligence_records
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2;
        """
        records = await conn.fetch(query, limit, offset)
        
        # Get total count for pagination
        count_query = "SELECT count(*) FROM intelligence_records;"
        total = await conn.fetchval(count_query)
        
        await conn.close()
        
        return {
            "status": "success",
            "data": [dict(r) for r in records],
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total,
                "total_pages": (total // limit) + (1 if total % limit > 0 else 0)
            }
        }
    except Exception as e:
        logger.error(f"Error fetching archive: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/v1/entities/strategic")
async def get_strategic_entities():
    """Fetches high-value, pre-seeded entities with coordinates to render as the 'Knowledge Underlay' on the globe."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        query = """
            SELECT 
                entity_id as uid, 
                name as headline, 
                entity_type as domain, 
                'KNOWLEDGE' as source_type,
                threat_score,
                ST_Y(primary_geo) as lat, 
                ST_X(primary_geo) as lon
            FROM entities
            WHERE primary_geo IS NOT NULL
            AND watch_status != 'PASSIVE'
            ORDER BY threat_score DESC
            LIMIT 500;
        """
        records = await conn.fetch(query)
        await conn.close()
        
        # We format them similarly to IntelligenceEvents so the UI can easily map them
        formatted = []
        for r in records:
            priority = 'NORMAL'
            if r['threat_score'] >= 0.8: priority = 'CRITICAL'
            elif r['threat_score'] >= 0.5: priority = 'HIGH'
            
            formatted.append({
                "uid": str(r['uid']),
                "headline": r['headline'],
                "domain": r['domain'],
                "source_type": r['source_type'],
                "priority": priority,
                "geo": {"lat": r['lat'], "lon": r['lon']}
            })
            
        return {"status": "success", "data": formatted}
    except Exception as e:
        logger.error(f"Error fetching strategic entities: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
