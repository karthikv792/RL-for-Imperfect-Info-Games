from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.game_session import GameSessionManager
from api.leaderboard import LeaderboardManager
from api.model_registry import ModelRegistry
from api.replay_store import ReplayStore

router = APIRouter()
manager = GameSessionManager()
leaderboard_manager = LeaderboardManager()
model_registry = ModelRegistry()
replay_store = ReplayStore()


class CreateGameRequest(BaseModel):
    agent: str = "random"
    seed: int | None = None
    human_player: int = 1


class MakeMoveRequest(BaseModel):
    position: list[int]
    type: str


@router.get("/agents")
def list_agents():
    """List all available AI agents."""
    return {"agents": manager.list_agents()}


@router.post("/game")
def create_game(req: CreateGameRequest):
    """Create a new game session against an AI agent."""
    try:
        sid = manager.create_session(
            agent_name=req.agent,
            seed=req.seed,
            human_player=req.human_player,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    session = manager.get_session(sid)
    return {"session_id": sid, "state": session.to_dict()}


@router.get("/game/{session_id}")
def get_game_state(session_id: str):
    """Retrieve the current state of a game session."""
    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return session.to_dict()


@router.post("/game/{session_id}/move")
def make_move(session_id: str, req: MakeMoveRequest):
    """Submit a human move and receive the AI agent's response move."""
    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    move_result = session.apply_human_move(req.position, req.type)
    response = {"move_result": move_result, "state": session.to_dict()}
    if move_result.get("success") and not session.is_over and not session.is_human_turn:
        agent_move = session.get_agent_move()
        response["agent_move"] = agent_move
        response["state"] = session.to_dict()
    return response


@router.get("/replays")
def list_replays(agent: str | None = None, limit: int = 20):
    """List game replays, optionally filtered by agent."""
    return replay_store.list_replays(limit=limit, agent=agent)


@router.get("/replays/{replay_id}")
def get_replay(replay_id: str):
    """Retrieve a specific game replay by ID."""
    replay = replay_store.get_replay(replay_id)
    if replay is None:
        raise HTTPException(status_code=404, detail="Replay not found")
    return replay


@router.get("/models")
def list_models():
    """List all registered models and their metadata."""
    return {"models": model_registry.list_models()}


@router.get("/leaderboard")
def get_leaderboard():
    """Retrieve the current agent leaderboard rankings."""
    return {"leaderboard": leaderboard_manager.get_leaderboard()}
