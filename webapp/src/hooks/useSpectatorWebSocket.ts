"use client";
import { useCallback, useEffect, useRef, useState } from "react";

interface CellData {
  card: string;
  occupant: number;
}

interface SpectatorState {
  board: CellData[][];
  move: { player: number; position: number[]; type: string } | null;
  moveCount: number;
  isOver: boolean;
  winner: number | null;
}

export function useSpectatorWebSocket(wsUrl: string) {
  const ws = useRef<WebSocket | null>(null);
  const [state, setState] = useState<SpectatorState | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [matchId, setMatchId] = useState<string | null>(null);
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
      if (data.type === "spectate_started") {
        setMatchId(data.match_id);
      } else if (data.type === "spectate_update") {
        setState({
          board: data.board,
          move: data.move,
          moveCount: data.move_count,
          isOver: data.is_over,
          winner: data.winner,
        });
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

  const startMatch = useCallback((agent1: string, agent2: string, seed?: number) => {
    ws.current?.send(JSON.stringify({ type: "spectate_start", agent1, agent2, seed }));
  }, []);

  const advance = useCallback(() => {
    if (matchId) {
      ws.current?.send(JSON.stringify({ type: "spectate_advance", match_id: matchId }));
    }
  }, [matchId]);

  return { state, isConnected, matchId, reconnectAttempt, startMatch, advance };
}
