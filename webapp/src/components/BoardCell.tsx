"use client";
import { Token } from "./Token";

interface BoardCellProps {
  card: string;
  occupant: number; // 0=empty, 1=player1, 2=player2, 3=corner
  isLegal: boolean;
  isLastMove: boolean;
  inSequence: boolean;
  onClick?: () => void;
  isHovered?: boolean;
}

export function BoardCell({
  card, occupant, isLegal, isLastMove, inSequence, onClick, isHovered,
}: BoardCellProps) {
  const isCorner = card === "XX";
  const suit = card.slice(-1);
  const rank = card.slice(0, -1);

  const suitSymbol: Record<string, string> = { S: "\u2660", H: "\u2665", D: "\u2666", C: "\u2663" };
  const suitColor = suit === "H" || suit === "D" ? "text-red-600" : "text-gray-900";

  const suitNames: Record<string, string> = { S: "Spades", H: "Hearts", D: "Diamonds", C: "Clubs" };
  const occupantLabel = occupant === 0 ? "empty" : occupant === 1 ? "Gold token" : occupant === 2 ? "Blue token" : "corner";
  const legalLabel = isLegal ? ", legal move available" : "";
  const ariaLabel = isCorner
    ? "Corner cell, free space"
    : `${rank} of ${suitNames[suit] || suit}, ${occupantLabel}${legalLabel}`;

  return (
    <div
      className={`relative aspect-square rounded-sm border transition-all duration-150
        ${isCorner ? "bg-gradient-to-br from-green-700 to-green-800 border-green-600" : "bg-[var(--surface-card)] border-[var(--border-card)]"}
        ${isLegal ? "cursor-pointer ring-2 ring-green-400/40 hover:ring-green-400/70 hover:scale-[1.02]" : ""}
        ${isLastMove ? "ring-2 ring-amber-400/50" : ""}
      `}
      role="gridcell"
      aria-label={ariaLabel}
      tabIndex={isLegal ? 0 : -1}
      onClick={isLegal ? onClick : undefined}
      onKeyDown={isLegal ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onClick?.(); } } : undefined}
    >
      {!isCorner && (
        <>
          <span className={`absolute top-0.5 left-1 text-[10px] font-semibold font-serif ${suitColor}`}>
            {rank}
          </span>
          <span className={`absolute bottom-0.5 right-1 text-[9px] font-serif ${suitColor}`}>
            {suitSymbol[suit] || ""}
          </span>
        </>
      )}
      {isCorner && (
        <span className="absolute inset-0 flex items-center justify-center text-green-300/60 text-lg">{"\u2605"}</span>
      )}
      {(occupant === 1 || occupant === 2) && (
        <Token player={occupant} inSequence={inSequence} />
      )}
    </div>
  );
}
