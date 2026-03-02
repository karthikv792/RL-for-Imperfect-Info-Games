# training/elo.py
from __future__ import annotations
import json


class EloRating:
    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        self.k = k_factor
        self.initial = initial_rating
        self.ratings: dict[str, float] = {}

    def add_player(self, name: str):
        if name not in self.ratings:
            self.ratings[name] = self.initial

    def get_rating(self, name: str) -> float:
        return self.ratings[name]

    def record_result(self, player_a: str, player_b: str, winner: str | None):
        ra = self.ratings[player_a]
        rb = self.ratings[player_b]
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400))
        eb = 1.0 - ea

        if winner == player_a:
            sa, sb = 1.0, 0.0
        elif winner == player_b:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        self.ratings[player_a] = ra + self.k * (sa - ea)
        self.ratings[player_b] = rb + self.k * (sb - eb)

    def leaderboard(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.ratings, f, indent=2)

    @classmethod
    def load(cls, path: str) -> EloRating:
        elo = cls()
        with open(path) as f:
            elo.ratings = json.load(f)
        return elo
