# training/arena.py
from __future__ import annotations
from engine.game_state import GameState
from agents.base import Agent


class Arena:
    def __init__(self, num_games: int = 100, max_moves: int = 500):
        self.num_games = num_games
        self.max_moves = max_moves

    def evaluate(
        self,
        player1: Agent,
        player2: Agent,
        base_seed: int = 0,
    ) -> dict[str, float | int]:
        p1_wins = 0
        p2_wins = 0
        draws = 0

        for game_idx in range(self.num_games):
            # Alternate who goes first
            if game_idx % 2 == 0:
                agents = {1: player1, 2: player2}
            else:
                agents = {1: player2, 2: player1}

            gs = GameState.new_game(seed=base_seed + game_idx)
            for _ in range(self.max_moves):
                if gs.is_terminal():
                    break
                current = gs.current_player
                info = gs.to_information_set(current)
                action = agents[current].select_action(gs, info)
                gs = gs.apply_action(action)

            winner = gs.get_winner()
            if winner is None:
                draws += 1
            elif agents[winner] is player1:
                p1_wins += 1
            else:
                p2_wins += 1

        return {
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "draws": draws,
            "p1_win_rate": p1_wins / self.num_games,
            "p2_win_rate": p2_wins / self.num_games,
        }
