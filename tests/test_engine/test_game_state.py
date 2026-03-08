import numpy as np
from engine.game_state import GameState
from engine.actions import ActionType
from engine.board import Occupant


class TestGameStateCreation:
    def test_new_game(self):
        gs = GameState.new_game(seed=42)
        assert gs.current_player == 1
        assert len(gs.hands[1]) == 5
        assert len(gs.hands[2]) == 5
        assert gs.deck_size > 0
        assert gs.discard_pile == ()
        assert not gs.is_terminal()

    def test_new_game_deterministic(self):
        gs1 = GameState.new_game(seed=42)
        gs2 = GameState.new_game(seed=42)
        assert gs1.hands[1] == gs2.hands[1]
        assert gs1.hands[2] == gs2.hands[2]


class TestApplyAction:
    def test_place_adds_token(self):
        gs = GameState.new_game(seed=42)
        legal = gs.get_legal_actions()
        place_actions = [a for a in legal if a.action_type == ActionType.PLACE]
        if place_actions:
            action = place_actions[0]
            new_gs = gs.apply_action(action)
            assert new_gs.occupancy[action.row][action.col] == Occupant.PLAYER1
            assert gs.occupancy[action.row][action.col] == Occupant.EMPTY

    def test_apply_action_switches_player(self):
        gs = GameState.new_game(seed=42)
        legal = gs.get_legal_actions()
        if legal:
            new_gs = gs.apply_action(legal[0])
            assert new_gs.current_player == 2

    def test_apply_action_draws_card(self):
        gs = GameState.new_game(seed=42)
        legal = gs.get_legal_actions()
        place_actions = [a for a in legal if a.action_type == ActionType.PLACE]
        if place_actions:
            old_hand_size = len(gs.hands[gs.current_player])
            new_gs = gs.apply_action(place_actions[0])
            assert len(new_gs.hands[1]) == old_hand_size

    def test_apply_action_adds_to_discard(self):
        gs = GameState.new_game(seed=42)
        legal = gs.get_legal_actions()
        if legal:
            new_gs = gs.apply_action(legal[0])
            assert len(new_gs.discard_pile) == 1


class TestTerminalState:
    def test_new_game_not_terminal(self):
        gs = GameState.new_game(seed=42)
        assert not gs.is_terminal()
        assert gs.get_winner() is None


class TestInformationSet:
    def test_info_set_hides_opponent_hand(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=1)
        assert info.own_hand == gs.hands[1]
        assert not hasattr(info, "opponent_hand")

    def test_info_set_preserves_public_info(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=1)
        assert info.current_player == gs.current_player
        assert info.deck_size == gs.deck_size
        np.testing.assert_array_equal(info.occupancy, gs.occupancy)


class TestTensorEncoding:
    def test_tensor_shape(self):
        gs = GameState.new_game(seed=42)
        tensor = gs.to_tensor(player_perspective=1)
        assert tensor.shape == (22, 10, 10)

    def test_tensor_dtype(self):
        gs = GameState.new_game(seed=42)
        tensor = gs.to_tensor(player_perspective=1)
        assert tensor.dtype == np.float32
