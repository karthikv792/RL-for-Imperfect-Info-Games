"use client";
import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Board } from "@/components/Board";
import { useReplayPlayer } from "@/hooks/useReplayPlayer";

interface Replay {
  id: string;
  agent1: string;
  agent2: string;
  winner: string | null;
  moves: { player: number; position: number[]; type: string }[];
  num_moves: number;
  created_at: string;
}

const SPEEDS = [
  { label: "0.5s", ms: 500 },
  { label: "1s", ms: 1000 },
  { label: "2s", ms: 2000 },
];

function ReplayViewer({ replay }: { replay: Replay }) {
  const {
    currentStep, totalSteps, isPlaying, board, lastMove,
    speed, setSpeed, play, pause, stepForward, stepBackward, goToStep, reset,
  } = useReplayPlayer(replay.moves);

  return (
    <div className="flex flex-col items-center gap-6">
      {/* Game Info */}
      <div className="bg-[var(--bg-card)] rounded-xl p-4 w-full max-w-lg">
        <div className="flex justify-between items-center">
          <div>
            <span className="text-amber-400 font-semibold">{replay.agent1}</span>
            <span className="text-[var(--text-muted)] mx-2">vs</span>
            <span className="text-blue-400 font-semibold">{replay.agent2}</span>
          </div>
          <div className="text-sm text-[var(--text-muted)]">
            {new Date(replay.created_at).toLocaleDateString()}
          </div>
        </div>
        {replay.winner && (
          <p className="text-sm text-[var(--text-secondary)] mt-1">
            Winner: <span className="font-semibold">{replay.winner}</span>
          </p>
        )}
      </div>

      {/* Board */}
      <div className="max-w-lg w-full">
        <Board
          board={board}
          legalMoves={[]}
          lastMove={lastMove}
          onCellClick={() => {}}
        />
      </div>

      {/* Timeline Slider */}
      <div className="w-full max-w-lg">
        <input
          type="range"
          min={0}
          max={totalSteps}
          value={currentStep}
          onChange={e => goToStep(Number(e.target.value))}
          className="w-full accent-amber-500"
        />
        <div className="flex justify-between text-sm text-[var(--text-muted)]">
          <span>Move {currentStep}</span>
          <span>{totalSteps} total</span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3">
        <button onClick={reset}
          className="px-3 py-2 bg-[var(--bg-card)] rounded-lg text-[var(--text-primary)] border border-gray-700 hover:border-gray-500 text-sm">
          Reset
        </button>
        <button onClick={stepBackward} disabled={currentStep === 0}
          className="px-3 py-2 bg-[var(--bg-card)] rounded-lg text-[var(--text-primary)] border border-gray-700 hover:border-gray-500 disabled:opacity-50 text-sm">
          Prev
        </button>
        <button onClick={isPlaying ? pause : play}
          className="px-4 py-2 bg-gradient-to-r from-amber-500 to-amber-600 text-white font-semibold rounded-lg hover:from-amber-400 hover:to-amber-500">
          {isPlaying ? "Pause" : "Play"}
        </button>
        <button onClick={stepForward} disabled={currentStep >= totalSteps}
          className="px-3 py-2 bg-[var(--bg-card)] rounded-lg text-[var(--text-primary)] border border-gray-700 hover:border-gray-500 disabled:opacity-50 text-sm">
          Next
        </button>
        <div className="flex gap-1 ml-2">
          {SPEEDS.map(s => (
            <button key={s.ms} onClick={() => setSpeed(s.ms)}
              className={`px-2 py-1 rounded text-xs ${speed === s.ms ? "bg-amber-500 text-white" : "bg-[var(--bg-card)] text-[var(--text-muted)]"}`}>
              {s.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function ReplayPage() {
  const params = useParams();
  const id = params.id as string;
  const [replay, setReplay] = useState<Replay | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const apiBase = typeof window !== "undefined"
      ? `http://${window.location.hostname}:8000`
      : "http://localhost:8000";

    fetch(`${apiBase}/api/replays/${id}`)
      .then(res => {
        if (!res.ok) throw new Error("Replay not found");
        return res.json();
      })
      .then(data => { setReplay(data); setLoading(false); })
      .catch(err => { setError(err.message); setLoading(false); });
  }, [id]);

  return (
    <main className="flex flex-col items-center min-h-screen px-4 py-8">
      <h1 className="text-3xl font-bold mb-6 text-[var(--text-primary)]">Game Replay</h1>

      {loading && <p className="text-[var(--text-muted)]">Loading replay...</p>}
      {error && <p className="text-red-400">{error}</p>}
      {replay && <ReplayViewer replay={replay} />}

      <Link href="/" className="mt-8 text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors text-sm">
        Back to Home
      </Link>
    </main>
  );
}
