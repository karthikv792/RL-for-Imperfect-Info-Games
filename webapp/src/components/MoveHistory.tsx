"use client";

interface Move {
  player: number;
  position: number[];
  type: string;
}

interface MoveHistoryProps {
  moves: Move[];
}

export function MoveHistory({ moves }: MoveHistoryProps) {
  return (
    <div className="bg-[var(--bg-card)] rounded-xl p-4 max-h-60 overflow-y-auto">
      <h3 className="text-sm font-semibold text-[var(--text-muted)] mb-2">Move History</h3>
      {moves.length === 0 ? (
        <p className="text-sm text-[var(--text-muted)]">No moves yet</p>
      ) : (
        <div className="space-y-1">
          {moves.map((move, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <span className={`w-2 h-2 rounded-full ${move.player === 1 ? "bg-amber-400" : "bg-blue-400"}`} />
              <span className="text-[var(--text-secondary)]">
                {move.player === 1 ? "You" : "AI"}: {move.type} at ({move.position[0]}, {move.position[1]})
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
