"use client";
import { useState, useCallback, useRef, useEffect } from "react";

interface Move {
  player: number;
  position: number[];
  type: string;
}

// Board layout for a 10x10 Sequence board — simplified, just need card labels
// For the replay viewer, we can use placeholder card values since
// the key visual is the occupant tokens
function createEmptyBoard(): { card: string; occupant: number }[][] {
  const board: { card: string; occupant: number }[][] = [];
  for (let r = 0; r < 10; r++) {
    const row: { card: string; occupant: number }[] = [];
    for (let c = 0; c < 10; c++) {
      // Corners are "free" spaces
      const isCorner = (r === 0 || r === 9) && (c === 0 || c === 9);
      row.push({ card: isCorner ? "FREE" : "", occupant: isCorner ? 0 : 0 });
    }
    board.push(row);
  }
  return board;
}

export function useReplayPlayer(moves: Move[]) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1000);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Reconstruct board state up to currentStep
  const board = createEmptyBoard();
  let lastMove: { position: number[] } | null = null;

  for (let i = 0; i < currentStep; i++) {
    const move = moves[i];
    if (move && move.position) {
      const [r, c] = move.position;
      if (move.type === "place") {
        board[r][c].occupant = move.player;
      } else if (move.type === "remove") {
        board[r][c].occupant = 0;
      }
      if (i === currentStep - 1) {
        lastMove = { position: move.position };
      }
    }
  }

  const play = useCallback(() => setIsPlaying(true), []);
  const pause = useCallback(() => setIsPlaying(false), []);
  const stepForward = useCallback(() => {
    setCurrentStep(s => Math.min(s + 1, moves.length));
  }, [moves.length]);
  const stepBackward = useCallback(() => {
    setCurrentStep(s => Math.max(s - 1, 0));
  }, []);
  const goToStep = useCallback((step: number) => {
    setCurrentStep(Math.max(0, Math.min(step, moves.length)));
  }, [moves.length]);
  const reset = useCallback(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, []);

  useEffect(() => {
    if (isPlaying && currentStep < moves.length) {
      intervalRef.current = setInterval(() => {
        setCurrentStep(s => {
          if (s >= moves.length) {
            setIsPlaying(false);
            return s;
          }
          return s + 1;
        });
      }, speed);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, currentStep, moves.length, speed]);

  return {
    currentStep,
    totalSteps: moves.length,
    isPlaying,
    board,
    lastMove,
    speed,
    setSpeed,
    play,
    pause,
    stepForward,
    stepBackward,
    goToStep,
    reset,
  };
}
