from __future__ import annotations
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from api.routes.game import router as game_router, manager
from api.ws_handler import handle_websocket, spectator_manager

app = FastAPI(
    title="Sequence AI",
    description="AI-powered Sequence board game with multiple state-of-the-art agents, real-time web play, and spectator mode.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(game_router, prefix="/api")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for real-time game interaction."""
    await handle_websocket(ws, manager)


@app.post("/api/spectate")
def spectate_start(agent1: str = "random", agent2: str = "heuristic", seed: int | None = None):
    """Start a new spectator match between two AI agents."""
    try:
        match_id = spectator_manager.start_match(agent1, agent2, seed=seed)
        return {"match_id": match_id}
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/spectate/{match_id}/advance")
def spectate_advance(match_id: str):
    """Advance a spectator match by one step."""
    result = spectator_manager.advance_match(match_id)
    return result


@app.get("/api/spectate")
def spectate_list():
    """List all active spectator matches."""
    return {"matches": spectator_manager.list_matches()}
