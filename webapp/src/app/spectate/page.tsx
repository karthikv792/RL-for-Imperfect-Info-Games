"use client";
import { useState, useEffect, useRef } from "react";
import { useSpectatorWebSocket } from "@/hooks/useSpectatorWebSocket";
import { Board } from "@/components/Board";

const AGENTS = [
  { id: "random", name: "Random" },
  { id: "heuristic", name: "Heuristic" },
];

const SPEEDS = [
  { label: "0.5s", ms: 500 },
  { label: "1s", ms: 1000 },
  { label: "2s", ms: 2000 },
];

export default function SpectatePage() {
  const [agent1, setAgent1] = useState("random");
  const [agent2, setAgent2] = useState("heuristic");
  const [speed, setSpeed] = useState(1000);
  const [isPlaying, setIsPlaying] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const wsUrl = typeof window !== "undefined"
    ? `ws://${window.location.hostname}:8000/ws`
    : "ws://localhost:8000/ws";

  const { state, isConnected, matchId, startMatch, advance } = useSpectatorWebSocket(wsUrl);

  const handleStart = () => {
    startMatch(agent1, agent2, Math.floor(Math.random() * 10000));
  };

  useEffect(() => {
    if (isPlaying && matchId && !state?.isOver) {
      intervalRef.current = setInterval(() => {
        advance();
      }, speed);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, matchId, speed, state?.isOver, advance]);

  useEffect(() => {
    if (state?.isOver) setIsPlaying(false);
  }, [state?.isOver]);

  return (
    <main className="flex flex-col items-center min-h-screen px-4 py-8">
      <h1 className="text-3xl font-bold mb-6 text-[var(--text-primary)]">Spectator Mode</h1>
      <p className="text-[var(--text-secondary)] mb-8">Watch AI agents battle it out</p>

      {!matchId ? (
        <div className="bg-[var(--bg-card)] rounded-xl p-6 space-y-4 max-w-md w-full">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm text-[var(--text-muted)] block mb-1">Agent 1 (Gold)</label>
              <select value={agent1} onChange={e => setAgent1(e.target.value)}
                className="w-full bg-[var(--bg-secondary)] text-[var(--text-primary)] p-2 rounded-lg border border-gray-700">
                {AGENTS.map(a => <option key={a.id} value={a.id}>{a.name}</option>)}
              </select>
            </div>
            <div>
              <label className="text-sm text-[var(--text-muted)] block mb-1">Agent 2 (Blue)</label>
              <select value={agent2} onChange={e => setAgent2(e.target.value)}
                className="w-full bg-[var(--bg-secondary)] text-[var(--text-primary)] p-2 rounded-lg border border-gray-700">
                {AGENTS.map(a => <option key={a.id} value={a.id}>{a.name}</option>)}
              </select>
            </div>
          </div>
          <button onClick={handleStart}
            className="w-full py-2 bg-gradient-to-r from-amber-500 to-amber-600 text-white font-semibold rounded-xl
              hover:from-amber-400 hover:to-amber-500 transition-all">
            Start Match
          </button>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-6">
          {/* Controls */}
          <div className="flex flex-wrap items-center justify-center gap-4">
            <button onClick={() => setIsPlaying(!isPlaying)}
              className="px-4 py-2 bg-[var(--bg-card)] rounded-lg text-[var(--text-primary)] border border-gray-700 hover:border-gray-500">
              {isPlaying ? "Pause" : "Play"}
            </button>
            <button onClick={advance} disabled={isPlaying}
              className="px-4 py-2 bg-[var(--bg-card)] rounded-lg text-[var(--text-primary)] border border-gray-700 hover:border-gray-500 disabled:opacity-50">
              Step
            </button>
            <div className="flex gap-1">
              {SPEEDS.map(s => (
                <button key={s.ms} onClick={() => setSpeed(s.ms)}
                  className={`px-3 py-1 rounded text-sm ${speed === s.ms ? "bg-amber-500 text-white" : "bg-[var(--bg-card)] text-[var(--text-muted)]"}`}>
                  {s.label}
                </button>
              ))}
            </div>
            <span className="text-sm text-[var(--text-muted)]">Move: {state?.moveCount || 0}</span>
          </div>

          {/* Board */}
          {state?.board && (
            <div className="max-w-lg w-full">
              <Board
                board={state.board}
                legalMoves={[]}
                lastMove={state.move ? { position: state.move.position } : null}
                onCellClick={() => {}}
              />
            </div>
          )}

          {/* Game Over */}
          {state?.isOver && (
            <div className="bg-[var(--bg-card)] rounded-xl p-4 text-center">
              <p className="text-lg font-semibold text-[var(--text-primary)]">
                {state.winner === 1 ? "Gold (Agent 1) Wins!" : state.winner === 2 ? "Blue (Agent 2) Wins!" : "Draw!"}
              </p>
              <p className="text-sm text-[var(--text-muted)]">in {state.moveCount} moves</p>
              <button onClick={() => window.location.reload()}
                className="mt-3 px-4 py-2 bg-amber-500 text-white rounded-lg hover:bg-amber-400">
                New Match
              </button>
            </div>
          )}
        </div>
      )}
    </main>
  );
}
