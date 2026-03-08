"use client";

interface HandCardProps {
  card: string;
  isSelected: boolean;
  isPlayable: boolean;
  onClick: () => void;
}

export function HandCard({ card, isSelected, isPlayable, onClick }: HandCardProps) {
  const suit = card.slice(-1);
  const rank = card.slice(0, -1);
  const suitSymbol: Record<string, string> = { S: "\u2660", H: "\u2665", D: "\u2666", C: "\u2663" };
  const suitColor = suit === "H" || suit === "D" ? "text-red-600" : "text-gray-900";

  const suitNames: Record<string, string> = { S: "Spades", H: "Hearts", D: "Diamonds", C: "Clubs" };
  const ariaLabel = `${rank} of ${suitNames[suit] || suit}${isSelected ? ", selected" : ""}${!isPlayable ? ", not playable" : ""}`;

  return (
    <button
      onClick={onClick}
      disabled={!isPlayable}
      aria-label={ariaLabel}
      aria-pressed={isSelected}
      className={`relative w-16 h-24 rounded-lg border-2 transition-all duration-200 bg-white
        ${isSelected ? "border-amber-500 -translate-y-3 shadow-lg shadow-amber-500/20" : "border-gray-300"}
        ${isPlayable ? "hover:-translate-y-1 hover:shadow-md cursor-pointer" : "opacity-50 cursor-not-allowed"}
      `}
    >
      <span className={`absolute top-1 left-1.5 text-sm font-bold ${suitColor}`}>{rank}</span>
      <span className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-2xl ${suitColor}`}>
        {suitSymbol[suit] || ""}
      </span>
      <span className={`absolute bottom-1 right-1.5 text-sm font-bold rotate-180 ${suitColor}`}>{rank}</span>
    </button>
  );
}
