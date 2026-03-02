import pytest
from engine.actions import Action, ActionType, action_to_index, index_to_action


class TestAction:
    def test_place_action_creation(self):
        a = Action(row=3, col=5, action_type=ActionType.PLACE)
        assert a.row == 3
        assert a.col == 5
        assert a.action_type == ActionType.PLACE

    def test_remove_action_creation(self):
        a = Action(row=7, col=2, action_type=ActionType.REMOVE)
        assert a.action_type == ActionType.REMOVE

    def test_wild_action_creation(self):
        a = Action(row=0, col=1, action_type=ActionType.WILD)
        assert a.action_type == ActionType.WILD


class TestActionEncoding:
    def test_place_to_index(self):
        a = Action(row=3, col=5, action_type=ActionType.PLACE)
        assert action_to_index(a) == 35

    def test_remove_to_index(self):
        a = Action(row=7, col=2, action_type=ActionType.REMOVE)
        assert action_to_index(a) == 172

    def test_wild_to_index(self):
        a = Action(row=0, col=1, action_type=ActionType.WILD)
        assert action_to_index(a) == 201

    def test_index_to_place(self):
        a = index_to_action(35)
        assert a.row == 3
        assert a.col == 5
        assert a.action_type == ActionType.PLACE

    def test_index_to_remove(self):
        a = index_to_action(172)
        assert a.row == 7
        assert a.col == 2
        assert a.action_type == ActionType.REMOVE

    def test_index_to_wild(self):
        a = index_to_action(201)
        assert a.row == 0
        assert a.col == 1
        assert a.action_type == ActionType.WILD

    def test_roundtrip_all_actions(self):
        for i in range(300):
            a = index_to_action(i)
            assert action_to_index(a) == i

    def test_invalid_index_raises(self):
        with pytest.raises(ValueError):
            index_to_action(300)
        with pytest.raises(ValueError):
            index_to_action(-1)

    def test_action_is_frozen(self):
        a = Action(row=0, col=0, action_type=ActionType.PLACE)
        with pytest.raises(AttributeError):
            a.row = 1
