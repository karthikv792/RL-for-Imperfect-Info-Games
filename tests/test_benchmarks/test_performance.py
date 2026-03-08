import pytest
from engine.game_state import GameState
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent


class TestPerformanceBenchmarks:
    def test_legal_actions_speed(self, benchmark):
        gs = GameState.new_game(seed=42)
        benchmark(gs.get_legal_actions)

    def test_apply_action_speed(self, benchmark):
        gs = GameState.new_game(seed=42)
        actions = gs.get_legal_actions()

        def apply_and_reset():
            gs.apply_action(actions[0])

        benchmark(apply_and_reset)

    def test_random_agent_game(self, benchmark):
        def play_game():
            gs = GameState.new_game(seed=42)
            agent = RandomAgent("r", seed=42)
            moves = 0
            while not gs.is_terminal() and moves < 200:
                info = gs.to_information_set(gs.current_player)
                action = agent.select_action(gs, info)
                gs = gs.apply_action(action)
                moves += 1

        benchmark(play_game)

    def test_heuristic_agent_game(self, benchmark):
        def play_game():
            gs = GameState.new_game(seed=42)
            agent = HeuristicAgent("h")
            moves = 0
            while not gs.is_terminal() and moves < 200:
                info = gs.to_information_set(gs.current_player)
                action = agent.select_action(gs, info)
                gs = gs.apply_action(action)
                moves += 1

        benchmark(play_game)

    def test_tokenizer_speed(self, benchmark):
        from models.board_tokenizer import BoardTokenizer
        gs = GameState.new_game(seed=42)
        tok = BoardTokenizer()
        benchmark(tok.tokenize, gs, player_perspective=1)
