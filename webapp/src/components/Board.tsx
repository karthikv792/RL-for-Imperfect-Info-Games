"use client";
import { BoardCell } from "./BoardCell";

interface CellData {
  card: string;
  occupant: number;
}

interface BoardProps {
  board: CellData[][];
  legalMoves: { position: number[]; type: string }[];
  lastMove?: { position: number[] } | null;
  sequences?: Record<string, number[][][]>;
  onCellClick?: (row: number, col: number) => void;
}

export function Board({ board, legalMoves, lastMove, sequences, onCellClick }: BoardProps) {
  const legalSet = new Set(legalMoves.map(m => `${m.position[0]},${m.position[1]}`));
  const lastMoveKey = lastMove ? `${lastMove.position[0]},${lastMove.position[1]}` : "";

  // Collect cells in sequences
  const sequenceCells = new Set<string>();
  if (sequences) {
    Object.values(sequences).forEach(seqs => {
      if (Array.isArray(seqs)) {
        seqs.forEach(seq => {
          if (Array.isArray(seq)) {
            seq.forEach(cell => {
              if (Array.isArray(cell)) sequenceCells.add(`${cell[0]},${cell[1]}`);
            });
          }
        });
      }
    });
  }

  return (
    <div className="w-full max-w-[min(90vw,500px)] bg-[var(--surface-board)] p-3 rounded-xl shadow-2xl">
      <div className="grid grid-cols-10 gap-0.5">
        {board.map((row, r) =>
          row.map((cell, c) => (
            <BoardCell
              key={`${r}-${c}`}
              card={cell.card}
              occupant={cell.occupant}
              isLegal={legalSet.has(`${r},${c}`)}
              isLastMove={`${r},${c}` === lastMoveKey}
              inSequence={sequenceCells.has(`${r},${c}`)}
              onClick={() => onCellClick?.(r, c)}
            />
          ))
        )}
      </div>
    </div>
  );
}
