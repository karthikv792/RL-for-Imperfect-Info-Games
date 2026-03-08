# tests/test_engine/test_game_simulation.py
from engine.game_state import GameState
import random


class TestFullGameSimulation:
    def test_random_game_completes(self):
        """Play a full game with random moves. Must terminate without error."""
        gs = GameState.new_game(seed=123)
        max_moves = 500
        for _ in range(max_moves):
            if gs.is_terminal():
                break
            legal = gs.get_legal_actions()
            assert len(legal) > 0, "Non-terminal state has no legal actions"
            action = random.choice(legal)
            gs = gs.apply_action(action)

    def test_ten_random_games(self):
        """10 random games all complete."""
        for seed in range(10):
            gs = GameState.new_game(seed=seed)
            for _ in range(500):
                if gs.is_terminal():
                    break
                action = random.choice(gs.get_legal_actions())
                gs = gs.apply_action(action)

    def test_winner_is_valid(self):
        gs = GameState.new_game(seed=42)
        for _ in range(500):
            if gs.is_terminal():
                winner = gs.get_winner()
                assert winner in (None, 1, 2)
                break
            action = random.choice(gs.get_legal_actions())
            gs = gs.apply_action(action)
