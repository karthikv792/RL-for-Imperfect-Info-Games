from __future__ import annotations
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from api.routes.game import router as game_router, manager
from api.ws_handler import handle_websocket

app = FastAPI(title="Sequence AI", version="0.2.0")

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
    await handle_websocket(ws, manager)
