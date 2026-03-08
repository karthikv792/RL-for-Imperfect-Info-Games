"use client";
import { useEffect, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Board } from "@/components/Board";
import { Hand } from "@/components/Hand";
import { GameInfo } from "@/components/GameInfo";
import { MoveHistory } from "@/components/MoveHistory";
import { useGameWebSocket } from "@/hooks/useGameWebSocket";
import { useGameState } from "@/hooks/useGameState";

function PlayContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const agentName = searchParams.get("agent") || "heuristic";

  const wsUrl = typeof window !== "undefined"
    ? `ws://${window.location.hostname}:8000/ws`
    : "ws://localhost:8000/ws";

  const { gameState, isConnected, aiThinking, newGame, makeMove } = useGameWebSocket(wsUrl);
  const { selectedCard, setSelectedCard, getPlayableCards, getLegalMovesForCard, resetSelection } = useGameState();

  // Start a new game when connected
  useEffect(() => {
    if (isConnected) {
      newGame(agentName);
    }
  }, [isConnected, agentName, newGame]);

  const handleCellClick = (row: number, col: number) => {
    if (!gameState || gameState.turn !== "human" || gameState.is_over) return;

    // Find the matching legal move for this cell
    const legalMove = gameState.legal_moves.find(
      m => m.position[0] === row && m.position[1] === col
    );
    if (legalMove) {
      makeMove([row, col], legalMove.type);
      resetSelection();
    }
  };

  const handleNewGame = () => {
    if (isConnected) {
      newGame(agentName);
      resetSelection();
    }
  };

  if (!gameState) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-[var(--text-muted)] text-lg">
          {isConnected ? "Starting game..." : "Connecting to server..."}
        </div>
      </div>
    );
  }

  const playableCards = getPlayableCards(gameState);

  return (
    <main className="flex flex-col items-center min-h-screen px-4 py-6">
      {/* Header */}
      <div className="flex items-center justify-between w-full max-w-5xl mb-4">
        <button
          onClick={() => router.push("/")}
          className="text-sm text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors"
        >
          &larr; Back
        </button>
        <h1 className="text-xl font-bold text-[var(--text-primary)]">
          vs {agentName.charAt(0).toUpperCase() + agentName.slice(1)}
        </h1>
        <button
          onClick={handleNewGame}
          className="text-sm px-3 py-1 bg-[var(--bg-card)] rounded-lg text-[var(--text-secondary)] hover:text-[var(--text-primary)] border border-gray-700"
        >
          New Game
        </button>
      </div>

      {/* Main layout */}
      <div className="flex gap-6 w-full max-w-5xl justify-center">
        {/* Left: Board + Hand */}
        <div className="flex flex-col items-center gap-4">
          <Board
            board={gameState.board}
            legalMoves={gameState.legal_moves}
            lastMove={gameState.move_history.length > 0 ? gameState.move_history[gameState.move_history.length - 1] : null}
            sequences={gameState.sequences}
            onCellClick={handleCellClick}
          />
          <Hand
            cards={gameState.hand}
            selectedCard={selectedCard}
            playableCards={playableCards}
            onSelectCard={setSelectedCard}
          />
        </div>

        {/* Right: Info + History */}
        <div className="flex flex-col gap-4 min-w-[220px]">
          <GameInfo
            turn={gameState.turn}
            currentPlayer={gameState.current_player}
            sequences={gameState.sequences}
            deckSize={gameState.deck_size}
            aiThinking={aiThinking}
            isOver={gameState.is_over}
            winner={gameState.winner}
          />
          <MoveHistory moves={gameState.move_history} />
        </div>
      </div>

      {/* Game Over Overlay */}
      {gameState.is_over && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
          <div className="bg-[var(--bg-card)] rounded-2xl p-8 text-center max-w-sm shadow-2xl">
            <h2 className="text-2xl font-bold mb-2 text-[var(--text-primary)]">
              {gameState.winner === 1 ? "You Win!" : gameState.winner === 2 ? "AI Wins!" : "Draw!"}
            </h2>
            <p className="text-[var(--text-muted)] mb-6">
              Game completed in {gameState.move_history.length} moves
            </p>
            <div className="flex gap-3 justify-center">
              <button
                onClick={handleNewGame}
                className="px-6 py-2 bg-gradient-to-r from-amber-500 to-amber-600 text-white font-semibold rounded-xl hover:from-amber-400 hover:to-amber-500"
              >
                Play Again
              </button>
              <button
                onClick={() => router.push("/")}
                className="px-6 py-2 bg-[var(--bg-secondary)] text-[var(--text-secondary)] rounded-xl border border-gray-700 hover:border-gray-500"
              >
                Home
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

export default function PlayPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-[var(--text-muted)] text-lg">Loading...</div>
      </div>
    }>
      <PlayContent />
    </Suspense>
  );
}
