"use client";
import { HandCard } from "./HandCard";

interface HandProps {
  cards: string[];
  selectedCard: string | null;
  playableCards: Set<string>;
  onSelectCard: (card: string) => void;
}

export function Hand({ cards, selectedCard, playableCards, onSelectCard }: HandProps) {
  return (
    <div className="flex gap-2 justify-center py-4">
      {cards.map((card, i) => (
        <HandCard
          key={`${card}-${i}`}
          card={card}
          isSelected={selectedCard === card}
          isPlayable={playableCards.has(card)}
          onClick={() => onSelectCard(card)}
        />
      ))}
    </div>
  );
}
