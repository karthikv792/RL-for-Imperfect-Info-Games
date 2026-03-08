# tests/test_agents/test_baselines.py
from engine.game_state import GameState
from agents.base import Agent
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent


class TestAgentProtocol:
    def test_random_agent_is_agent(self):
        agent = RandomAgent(name="random")
        assert isinstance(agent, Agent)

    def test_heuristic_agent_is_agent(self):
        agent = HeuristicAgent(name="heuristic")
        assert isinstance(agent, Agent)


class TestRandomAgent:
    def test_selects_legal_action(self):
        gs = GameState.new_game(seed=42)
        agent = RandomAgent(name="random", seed=42)
        info = gs.to_information_set(player=gs.current_player)
        action = agent.select_action(gs, info)
        assert action in gs.get_legal_actions()

    def test_plays_full_game(self):
        gs = GameState.new_game(seed=42)
        agents = {1: RandomAgent("r1", seed=1), 2: RandomAgent("r2", seed=2)}
        for _ in range(500):
            if gs.is_terminal():
                break
            info = gs.to_information_set(gs.current_player)
            action = agents[gs.current_player].select_action(gs, info)
            gs = gs.apply_action(action)
        assert gs.is_terminal()


class TestHeuristicAgent:
    def test_selects_legal_action(self):
        gs = GameState.new_game(seed=42)
        agent = HeuristicAgent(name="heuristic")
        info = gs.to_information_set(player=gs.current_player)
        action = agent.select_action(gs, info)
        assert action in gs.get_legal_actions()

    def test_plays_full_game(self):
        gs = GameState.new_game(seed=42)
        agents = {1: HeuristicAgent("h1"), 2: HeuristicAgent("h2")}
        for _ in range(500):
            if gs.is_terminal():
                break
            info = gs.to_information_set(gs.current_player)
            action = agents[gs.current_player].select_action(gs, info)
            gs = gs.apply_action(action)
        assert gs.is_terminal()
