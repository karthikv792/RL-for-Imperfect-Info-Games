from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.game_session import GameSessionManager

router = APIRouter()
manager = GameSessionManager()


class CreateGameRequest(BaseModel):
    agent: str = "random"
    seed: int | None = None
    human_player: int = 1


class MakeMoveRequest(BaseModel):
    position: list[int]
    type: str


@router.get("/agents")
def list_agents():
    return {"agents": manager.list_agents()}


@router.post("/game")
def create_game(req: CreateGameRequest):
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
    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return session.to_dict()


@router.post("/game/{session_id}/move")
def make_move(session_id: str, req: MakeMoveRequest):
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


@router.get("/leaderboard")
def get_leaderboard():
    from training.elo import EloRating
    try:
        elo = EloRating.load("data/elo_ratings.json")
        return {"leaderboard": elo.leaderboard()}
    except FileNotFoundError:
        return {"leaderboard": []}
