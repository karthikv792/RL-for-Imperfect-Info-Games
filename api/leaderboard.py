from __future__ import annotations
from training.elo import EloRating


class LeaderboardManager:
    """Tracks agent performance across games."""

    def __init__(self):
        self.elo = EloRating()
        self.results: list[dict] = []

    def record_result(self, agent1: str, agent2: str, winner: str | None) -> None:
        self.results.append({"agent1": agent1, "agent2": agent2, "winner": winner})
        # Ensure both players are registered before recording
        self.elo.add_player(agent1)
        self.elo.add_player(agent2)
        self.elo.record_result(agent1, agent2, winner)

    def get_stats(self) -> dict[str, dict]:
        stats: dict[str, dict] = {}
        for r in self.results:
            for agent in [r["agent1"], r["agent2"]]:
                if agent not in stats:
                    stats[agent] = {"wins": 0, "losses": 0, "draws": 0, "games": 0}
                stats[agent]["games"] += 1
            if r["winner"]:
                stats[r["winner"]]["wins"] += 1
                loser = r["agent2"] if r["winner"] == r["agent1"] else r["agent1"]
                stats[loser]["losses"] += 1
            else:
                stats[r["agent1"]]["draws"] += 1
                stats[r["agent2"]]["draws"] += 1

        for agent in stats:
            stats[agent]["elo"] = self.elo.get_rating(agent)

        return stats

    def get_head_to_head(self, agent1: str, agent2: str) -> dict[str, int]:
        h2h = {agent1: 0, agent2: 0, "draws": 0}
        for r in self.results:
            if set([r["agent1"], r["agent2"]]) == set([agent1, agent2]):
                if r["winner"] == agent1:
                    h2h[agent1] += 1
                elif r["winner"] == agent2:
                    h2h[agent2] += 1
                else:
                    h2h["draws"] += 1
        return h2h

    def get_leaderboard(self) -> list[dict]:
        stats = self.get_stats()
        return sorted(
            [{"agent": k, **v} for k, v in stats.items()],
            key=lambda x: x["elo"],
            reverse=True,
        )
