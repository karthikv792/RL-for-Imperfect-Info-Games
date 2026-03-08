"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { GameState, AgentMove } from "@/types/game";

export function useGameWebSocket(wsUrl: string) {
  const ws = useRef<WebSocket | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [aiThinking, setAiThinking] = useState(false);
  const [lastAgentMove, setLastAgentMove] = useState<AgentMove | null>(null);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const reconnectTimer = useRef<NodeJS.Timeout | null>(null);
  const maxReconnectDelay = 30000;

  const connect = useCallback(() => {
    const socket = new WebSocket(wsUrl);
    ws.current = socket;

    socket.onopen = () => {
      setIsConnected(true);
      setReconnectAttempt(0);
    };

    socket.onclose = () => {
      setIsConnected(false);
      // Schedule reconnect with exponential backoff
      const delay = Math.min(1000 * Math.pow(2, reconnectAttempt), maxReconnectDelay);
      reconnectTimer.current = setTimeout(() => {
        setReconnectAttempt(prev => prev + 1);
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
  }, [wsUrl, reconnectAttempt]);

  useEffect(() => {
    connect();

    return () => {
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

  return { gameState, isConnected, aiThinking, lastAgentMove, reconnectAttempt, newGame, makeMove };
}
