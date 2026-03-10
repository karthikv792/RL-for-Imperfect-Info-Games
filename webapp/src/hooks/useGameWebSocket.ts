"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { GameState, AgentMove } from "@/types/game";

export function useGameWebSocket(wsUrl: string) {
  const ws = useRef<WebSocket | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [aiThinking, setAiThinking] = useState(false);
  const [lastAgentMove, setLastAgentMove] = useState<AgentMove | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimer = useRef<NodeJS.Timeout | null>(null);
  const maxReconnectDelay = 30000;
  const closedIntentionally = useRef(false);

  const connect = useCallback(() => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) return;

    const socket = new WebSocket(wsUrl);
    ws.current = socket;

    socket.onopen = () => {
      setIsConnected(true);
      reconnectAttemptRef.current = 0;
    };

    socket.onclose = () => {
      setIsConnected(false);
      if (closedIntentionally.current) return;
      const delay = Math.min(1000 * Math.pow(2, reconnectAttemptRef.current), maxReconnectDelay);
      reconnectAttemptRef.current += 1;
      reconnectTimer.current = setTimeout(() => {
        connect();
      }, delay);
    };

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
  }, [wsUrl]);

  useEffect(() => {
    closedIntentionally.current = false;
    connect();

    return () => {
      closedIntentionally.current = true;
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
      }
      ws.current?.close();
    };
  }, [connect]);

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
