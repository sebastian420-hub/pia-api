import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import List

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

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
        query = "SELECT cluster_id, name, status, confidence FROM intelligence_clusters WHERE status = 'ACTIVE' LIMIT 50;"
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

@app.get("/api/v1/graph/network/{entity_name}")
async def get_entity_network(entity_name: str, hops: int = 1):
    """Fetches the relational network for a specific entity to render in a 3D Force Graph."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        # 1. Find the root entity
        root = await conn.fetchrow("SELECT entity_id, name, entity_type FROM entities WHERE name ILIKE $1 LIMIT 1", entity_name)
        if not root:
            return {"status": "error", "message": f"Entity '{entity_name}' not found in the Knowledge Graph."}
            
        # 2. Find connected entities (1 hop for now to prevent massive payloads)
        query = """
            SELECT 
                r.relationship_id,
                r.relationship_type,
                r.confidence,
                e_a.entity_id as source_id,
                e_a.name as source_name,
                e_a.entity_type as source_type,
                e_b.entity_id as target_id,
                e_b.name as target_name,
                e_b.entity_type as target_type
            FROM entity_relationships r
            JOIN entities e_a ON r.entity_a_id = e_a.entity_id
            JOIN entities e_b ON r.entity_b_id = e_b.entity_id
            WHERE r.entity_a_id = $1 OR r.entity_b_id = $1
            LIMIT 100;
        """
        edges = await conn.fetch(query, root['entity_id'])
        
        nodes_dict = {}
        links = []
        
        # Always add the root node
        nodes_dict[str(root['entity_id'])] = {
            "id": str(root['entity_id']),
            "name": root['name'],
            "group": root['entity_type'],
            "val": 20 # Root node is larger
        }
        
        for edge in edges:
            s_id = str(edge['source_id'])
            t_id = str(edge['target_id'])
            
            # Add Source Node
            if s_id not in nodes_dict:
                nodes_dict[s_id] = {"id": s_id, "name": edge['source_name'], "group": edge['source_type'], "val": 5}
            
            # Add Target Node
            if t_id not in nodes_dict:
                nodes_dict[t_id] = {"id": t_id, "name": edge['target_name'], "group": edge['target_type'], "val": 5}
                
            # Add Link
            links.append({
                "source": s_id,
                "target": t_id,
                "label": edge['relationship_type'],
                "confidence": edge['confidence']
            })
            
        await conn.close()
        
        return {
            "status": "success",
            "data": {
                "nodes": list(nodes_dict.values()),
                "links": links
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching graph network: {e}")
        return {"status": "error", "message": str(e)}

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
    """Fetches the latest agent activity from the analysis queue to display in the UI Terminal."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        query = """
            SELECT 
                created_at, 
                status, 
                assigned_agent, 
                trigger_type, 
                priority,
                error_message
            FROM analysis_queue 
            ORDER BY created_at DESC 
            LIMIT 30;
        """
        records = await conn.fetch(query)
        await conn.close()
        
        logs = []
        for r in records:
            time_str = r['created_at'].strftime("%H:%M:%S")
            agent = r['assigned_agent'] or 'SYSTEM'
            if r['status'] == 'FAILED':
                msg = f"[{time_str}] [{agent}] ERROR: {r['error_message']}"
            elif r['status'] == 'PROCESSING':
                msg = f"[{time_str}] [{agent}] PROCESSING: {r['trigger_type']} ({r['priority']})"
            else:
                msg = f"[{time_str}] [{agent}] COMPLETED: {r['trigger_type']}"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
