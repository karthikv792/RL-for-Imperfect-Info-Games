"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { GameState, AgentMove } from "@/types/game";

export function useGameWebSocket(wsUrl: string) {
  const ws = useRef<WebSocket | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [aiThinking, setAiThinking] = useState(false);
  const [lastAgentMove, setLastAgentMove] = useState<AgentMove | null>(null);

  useEffect(() => {
    const socket = new WebSocket(wsUrl);
    ws.current = socket;

    socket.onopen = () => setIsConnected(true);
    socket.onclose = () => setIsConnected(false);

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "game_state") {
        setGameState(data);
        setAiThinking(false);
      } else if (data.type === "agent_move") {
        setLastAgentMove(data);
      } else if (data.type === "game_over") {
        setGameState(data);
        setAiThinking(false);
      }
    };

    return () => socket.close();
  }, [wsUrl]);

  const newGame = useCallback((agent: string, seed?: number) => {
    ws.current?.send(JSON.stringify({ type: "new_game", agent, seed }));
  }, []);

  const makeMove = useCallback((position: number[], moveType: string) => {
    ws.current?.send(JSON.stringify({
      type: "make_move", position, move_type: moveType,
    }));
    setAiThinking(true);
  }, []);

  return { gameState, isConnected, aiThinking, lastAgentMove, newGame, makeMove };
}
