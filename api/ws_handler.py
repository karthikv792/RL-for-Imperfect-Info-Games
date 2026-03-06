from __future__ import annotations
import json
from fastapi import WebSocket, WebSocketDisconnect
from api.game_session import GameSession, GameSessionManager, AVAILABLE_AGENTS
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent


async def handle_websocket(ws: WebSocket, manager: GameSessionManager):
    """Handle a WebSocket connection for a single player."""
    await ws.accept()
    session: GameSession | None = None

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "new_game":
                agent_name = msg.get("agent", "random")
                seed = msg.get("seed")
                try:
                    sid = manager.create_session(agent_name=agent_name, seed=seed)
                    session = manager.get_session(sid)
                    await ws.send_json({
                        "type": "game_state",
                        **session.to_dict(),
                    })
                except ValueError as e:
                    await ws.send_json({"type": "error", "message": str(e)})

            elif msg_type == "make_move":
                if session is None:
                    await ws.send_json({"type": "error", "message": "No active game"})
                    continue

                position = msg.get("position")
                move_type = msg.get("move_type")
                result = session.apply_human_move(position, move_type)

                if not result.get("success"):
                    await ws.send_json({
                        "type": "error",
                        "message": result.get("error", "Invalid move"),
                    })
                    continue

                if session.is_over:
                    await ws.send_json({
                        "type": "game_over",
                        "winner": session.game_state.get_winner(),
                        **session.to_dict(),
                    })
                    continue

                if not session.is_human_turn:
                    agent_result = session.get_agent_move()
                    await ws.send_json({
                        "type": "agent_move",
                        **agent_result,
                    })

                    if session.is_over:
                        await ws.send_json({
                            "type": "game_over",
                            "winner": session.game_state.get_winner(),
                            **session.to_dict(),
                        })
                    else:
                        await ws.send_json({
                            "type": "game_state",
                            **session.to_dict(),
                        })

            else:
                await ws.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        pass
