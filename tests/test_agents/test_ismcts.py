# tests/test_agents/test_ismcts.py
import numpy as np
from engine.game_state import GameState
from agents.ismcts.ismcts import ismcts_search, determinize


class TestDeterminize:
    def test_returns_valid_game_state(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=1)
        det = determinize(info, player=1, seed=99)
        assert isinstance(det, GameState)
        assert det.current_player == info.current_player

    def test_preserves_own_hand(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=1)
        det = determinize(info, player=1, seed=99)
        assert det.hands[1] == info.own_hand

    def test_opponent_hand_correct_size(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=1)
        det = determinize(info, player=1, seed=99)
        # Opponent hand should have the same size as own hand
        assert len(det.hands[2]) == len(info.own_hand)

    def test_total_card_count_consistent(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=1)
        det = determinize(info, player=1, seed=99)
        # Total cards across hands + deck + discard should equal
        # total deck minus cards placed on board
        total = (
            len(det.hands[1])
            + len(det.hands[2])
            + len(det.deck_cards)
            + len(det.discard_pile)
        )
        from engine.deck import SEQUENCE_DECK
        from engine.board import Occupant
        placed = 0
        for r in range(10):
            for c in range(10):
                if info.occupancy[r][c] in (Occupant.PLAYER1, Occupant.PLAYER2):
                    placed += 1
        assert total == len(SEQUENCE_DECK) - placed

    def test_determinize_different_seeds_give_different_hands(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=1)
        det1 = determinize(info, player=1, seed=1)
        det2 = determinize(info, player=1, seed=2)
        # Different seeds should usually give different opponent hands
        assert det1.hands[2] != det2.hands[2]


class TestISMCTSSearch:
    def test_returns_valid_action(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=gs.current_player)

        def dummy_policy_value(state):
            return np.ones(300) / 300, 0.0

        action = ismcts_search(
            info_set=info,
            player=gs.current_player,
            policy_value_fn=dummy_policy_value,
            num_determinizations=3,
            num_simulations=5,
            cpuct=1.5,
        )
        assert action in gs.get_legal_actions()
