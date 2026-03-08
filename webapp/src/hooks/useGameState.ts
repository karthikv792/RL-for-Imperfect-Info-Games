"use client";
import { useState, useCallback } from "react";
import type { GameState, LegalMove } from "@/types/game";

export function useGameState() {
  const [selectedCard, setSelectedCard] = useState<string | null>(null);

  const getPlayableCards = useCallback((gameState: GameState | null): Set<string> => {
    if (!gameState || gameState.turn !== "human") return new Set();
    const playable = new Set<string>();
    for (const card of gameState.hand) {
      playable.add(card);
    }
    return playable;
  }, []);

  const getLegalMovesForCard = useCallback((gameState: GameState | null, card: string | null): LegalMove[] => {
    if (!gameState || !card) return [];
    return gameState.legal_moves;
  }, []);

  const resetSelection = useCallback(() => {
    setSelectedCard(null);
  }, []);

  return { selectedCard, setSelectedCard, getPlayableCards, getLegalMovesForCard, resetSelection };
}
