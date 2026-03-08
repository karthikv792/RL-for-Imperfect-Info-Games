"use client";

interface GameInfoProps {
  turn: "human" | "ai";
  currentPlayer: number;
  sequences: Record<string, number[][][]>;
  deckSize: number;
  aiThinking: boolean;
  isOver: boolean;
  winner: number | null;
}

export function GameInfo({ turn, currentPlayer, sequences, deckSize, aiThinking, isOver, winner }: GameInfoProps) {
  const p1Seqs = sequences?.["1"]?.length || 0;
  const p2Seqs = sequences?.["2"]?.length || 0;

  return (
    <div className="bg-[var(--bg-card)] rounded-xl p-4 space-y-4 min-w-[200px]">
      <h2 className="text-lg font-semibold text-[var(--text-primary)]">Game Info</h2>

      {/* Turn indicator */}
      <div className="space-y-1" aria-live="polite">
        <div className="text-sm text-[var(--text-muted)]">Turn</div>
        <div className={`text-sm font-medium ${turn === "human" ? "text-green-400" : "text-amber-400"}`}>
          {aiThinking ? (
            <span className="flex items-center gap-2">
              <span className="w-2 h-2 bg-amber-400 rounded-full animate-pulse" />
              AI Thinking...
            </span>
          ) : turn === "human" ? "Your Turn" : "AI's Turn"}
        </div>
      </div>

      {/* Sequences */}
      <div className="space-y-2">
        <div className="text-sm text-[var(--text-muted)]">Sequences</div>
        <div className="flex justify-between text-sm">
          <span className="text-amber-400">You: {p1Seqs}/2</span>
          <span className="text-blue-400">AI: {p2Seqs}/2</span>
        </div>
        <div className="flex gap-1">
          {[0, 1].map(i => (
            <div key={`p1-${i}`} className={`h-1.5 flex-1 rounded-full ${i < p1Seqs ? "bg-amber-400" : "bg-gray-700"}`} />
          ))}
          {[0, 1].map(i => (
            <div key={`p2-${i}`} className={`h-1.5 flex-1 rounded-full ${i < p2Seqs ? "bg-blue-400" : "bg-gray-700"}`} />
          ))}
        </div>
      </div>

      {/* Deck */}
      <div className="space-y-1">
        <div className="text-sm text-[var(--text-muted)]">Deck</div>
        <div className="text-sm text-[var(--text-primary)]">{deckSize} cards remaining</div>
      </div>

      {/* Game Over */}
      {isOver && (
        <div className={`p-3 rounded-lg text-center font-semibold ${
          winner === 1 ? "bg-green-500/20 text-green-400" : winner === 2 ? "bg-red-500/20 text-red-400" : "bg-gray-500/20 text-gray-400"
        }`}>
          {winner === 1 ? "You Win!" : winner === 2 ? "AI Wins!" : "Draw!"}
        </div>
      )}
    </div>
  );
}
