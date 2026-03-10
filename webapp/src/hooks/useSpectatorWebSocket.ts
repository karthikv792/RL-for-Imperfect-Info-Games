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

  const startMatch = useCallback((agent1: string, agent2: string, seed?: number) => {
    ws.current?.send(JSON.stringify({ type: "spectate_start", agent1, agent2, seed }));
  }, []);

  const advance = useCallback(() => {
    if (matchId) {
      ws.current?.send(JSON.stringify({ type: "spectate_advance", match_id: matchId }));
    }
  }, [matchId]);

  return { state, isConnected, matchId, startMatch, advance };
}
