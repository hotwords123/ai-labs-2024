from util import *

import numpy as np
from pathlib import Path
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import re
from dataclasses import dataclass

result_pattern = re.compile(r'Win: (\d+), Lose: (\d+), Draw: (\d+)')

class EloRatingModel(nn.Module):
    def __init__(self, num_players: int):
        super().__init__()
        self.ratings = nn.Parameter(torch.zeros(num_players, dtype=torch.float32))
        self.ratings.data.fill_(1000)

        self.s = np.log(10) / 800
        self.k = nn.Parameter(torch.tensor(0, dtype=torch.float32))
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

    @staticmethod
    def _parse_stats(text: str) -> PlayerStats:
        match = result_pattern.search(text)
        win, lose, draw = map(int, match.groups())
        return PlayerStats(win, lose, draw)
    
    def _update_stats(self, player1: str, player2: str, stats: PlayerStats):
        self.players[player1].stats += stats
        self.players[player2].stats += stats.inverse()

        for score, count in [(1, stats.n_win), (-1, stats.n_lose), (0, stats.n_draw)]:
            for _ in range(count):
                self.results.append(PitResult(player1, player2, score))

    def load_results(self, result_dir: Path):
        for dir in result_dir.iterdir():
            if not dir.is_dir():
                continue

            player1, player2 = dir.name.split('_')

            for player in (player1, player2):
                if player not in self.players:
                    self.players[player] = Player(len(self.players), PlayerStats())

            lines = (dir / 'stdout.log').read_text().splitlines()
            first_play = self._parse_stats(lines[1])
            second_play = self._parse_stats(lines[2])

            self._update_stats(player1, player2, first_play)
            self._update_stats(player2, player1, second_play.inverse())

    def elo_rating(self, k: int = 16, seed: int | None = None) -> dict[str, int]:
        ratings = {player: 1000 for player in self.players}

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
        step_size: int = 30,
        gamma: float = 0.9,
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


if __name__ == '__main__':
    elo = EloRating()
    elo.load_results(Path('results/pit'))

    print("#players:", len(elo.players))
    print("#results:", len(elo.results))

    print("================================")

    # ratings = elo.elo_rating()
    ratings, model = elo.elo_rating_sgd()

    baseline_player = "train-50"
    baseline_rating = 1000
    delta = ratings[baseline_player] - baseline_rating

    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for player, rating in sorted_ratings:
        print(f'{player}: {rating - delta:.0f} ({elo.players[player].stats})')

    data = []
    for player, rating in ratings.items():
        x = int(player.removeprefix('train-'))
        y = rating - delta
        data.append((x, y))

    df = pd.DataFrame(data, columns=['x', 'y'])
    df.sort_values('x', inplace=True)

    plt.plot(df['x'], df['y'], marker='o')
    plt.xlabel('Train Step')
    plt.ylabel('Elo Rating')
    plt.title('Elo Rating of AlphaZero')
    plt.tight_layout()
    plt.savefig('results/elo_rating.png')

    # while True:
    #     p1, p2 = (
    #         torch.tensor([elo.players[p].id], dtype=torch.long)
    #         for p in input().split()
    #     )
    #     out = model((p1, p2))
    #     out = F.softmax(out, dim=1).detach().numpy()[0]
    #     print(f'p1 win: {out[0]:.3f}, draw: {out[1]:.3f}, p2 win: {out[2]:.3f}')
