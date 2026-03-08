"use client";

interface TokenProps {
  player: number;
  inSequence?: boolean;
  animate?: boolean;
}

export function Token({ player, inSequence = false, animate = true }: TokenProps) {
  const isGold = player === 1;
  return (
    <div
      className={`absolute top-[15%] left-[15%] w-[70%] h-[70%] rounded-full border-2
        ${animate ? "animate-token-place" : ""}
        ${isGold
          ? "bg-gradient-to-br from-amber-300 to-amber-500 border-amber-600"
          : "bg-gradient-to-br from-blue-300 to-blue-500 border-blue-600"
        }
      `}
      style={{
        boxShadow: `inset 0 2px 4px rgba(255,255,255,0.3), inset 0 -2px 4px rgba(0,0,0,0.15)${
          inSequence ? `, 0 0 12px ${isGold ? "rgba(245,158,11,0.4)" : "rgba(59,130,246,0.4)"}` : ""
        }`,
      }}
    />
  );
}
