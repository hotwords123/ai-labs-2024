from util import *

import numpy as np
from pathlib import Path
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataclasses import dataclass
from sgf import SGFParser


def parse_pit_sgf(content: str) -> tuple[PlayerStats, PlayerStats]:
    """
    Parse the result of pit games from an SGF file.
    The stats are from the perspective of the player who plays as black.

    Args:
        content: The content of the SGF file.

    Returns:
        black: The stats where the first player is black.
        white: The stats where the first player is white.
    """
    sgf = SGFParser(content).parse()
    stats = {k: PlayerStats() for k in ("black", "white")}

    for game in sgf.games:
        # RE is "B+{score}" or "W+{score}" or "Draw"
        re = game.root.get("RE")
        if re.startswith("B+"):
            score = 1
        elif re.startswith("W+"):
            score = -1
        elif re == "Draw":
            score = 0
        else:
            raise ValueError(f"Invalid result: {re}")

        # GN is "pit_{role}_{datetime}"
        role = game.root.get("GN").split("_")[1]
        stats[role].update(score)

    return stats["black"], stats["white"]


class EloRatingModel(nn.Module):
    def __init__(self, num_players: int):
        super().__init__()
        self.ratings = nn.Parameter(torch.zeros(num_players, dtype=torch.float32))

        # Scale factor
        self.s = np.log(10) / 800
        # Draw margin
        self.k = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        # First player advantage
        self.b = nn.Parameter(torch.tensor(0, dtype=torch.float32))

    def forward(self, x) -> torch.Tensor:
        r1 = self.ratings[x[0]]
        r2 = self.ratings[x[1]]

        p1_win = self.s * (r1 - r2) + self.b
        p2_win = self.s * (r2 - r1) - self.b
        draw = torch.zeros_like(p1_win) + self.k
        return torch.stack([p1_win, draw, p2_win], dim=1)


@dataclass
class Player:
    id: int
    stats: PlayerStats


class PitResult(NamedTuple):
    player1: str
    player2: str
    score: int


class EloRating:
    def __init__(self):
        self.players: dict[str, Player] = {}
        self.results: list[PitResult] = []

    def _update_stats(self, player1: str, player2: str, stats: PlayerStats):
        self.players[player1].stats += stats
        self.players[player2].stats += stats.inverse()

        for score, count in [(1, stats.n_win), (-1, stats.n_lose), (0, stats.n_draw)]:
            for _ in range(count):
                self.results.append(PitResult(player1, player2, score))

    def load_results(self, result_dir: str):
        dirs = [d for d in Path(result_dir).iterdir() if d.is_dir()]

        for dir in tqdm(dirs, desc='Loading results'):
            player1, player2 = dir.name.split('_vs_')

            for player in (player1, player2):
                if player not in self.players:
                    self.players[player] = Player(len(self.players), PlayerStats())

            sgf = (dir / 'match.sgf').read_text()
            black, white = parse_pit_sgf(sgf)
            self._update_stats(player1, player2, black)
            self._update_stats(player2, player1, white)

    def elo_rating(self, k: int = 16, seed: int | None = None) -> dict[str, int]:
        ratings = {player: 0 for player in self.players}

        results = self.results.copy()
        np.random.default_rng(seed).shuffle(results)

        for player1, player2, score in results:
            expected = 1 / (1 + 10 ** ((ratings[player2] - ratings[player1]) / 400))
            ratings[player1] += k * ((1 + score) / 2 - expected)
            ratings[player2] += k * ((1 - score) / 2 - expected)

        return ratings

    def elo_rating_sgd(
        self,
        lr: float = 10.0,
        num_iter: int = 1000,
        step_size: int = 25,
        gamma: float = 0.95,
    ) -> tuple[dict[str, int], EloRatingModel,]:
        model = EloRatingModel(len(self.players))
        model.train()

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        p1 = torch.tensor([self.players[r.player1].id for r in self.results], dtype=torch.long)
        p2 = torch.tensor([self.players[r.player2].id for r in self.results], dtype=torch.long)
        target = torch.tensor([1 - r.score for r in self.results], dtype=torch.long)

        progress_bar = tqdm(range(num_iter), desc='Elo Rating SGD', leave=False)
        for _ in progress_bar:
            optimizer.zero_grad()
            out = model((p1, p2))
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            scheduler.step()
            progress_bar.set_postfix(loss=loss.item())

        ratings = model.ratings.detach().numpy()
        ratings = {player: ratings[data.id] for player, data in self.players.items()}
        return ratings, model
