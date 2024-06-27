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
from dataclasses import dataclass
from sgf import SGFParser

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

    def _add_result(self, player1: str, player2: str, score: int):
        self.players[player1].stats.update(score)
        self.players[player2].stats.update(-score)
        self.results.append(PitResult(player1, player2, score))

    def load_results(self, result_dir: str):
        for dir in Path(result_dir).iterdir():
            if not dir.is_dir():
                continue

            player1, player2 = dir.name.split('_vs_')

            for player in (player1, player2):
                if player not in self.players:
                    self.players[player] = Player(len(self.players), PlayerStats())

            sgf = (dir / 'match.sgf').read_text()
            sgf = SGFParser(sgf).parse()

            for game in sgf.games:
                result = game.root.get("RE")
                if result.startswith("B+"):
                    score = 1
                elif result.startswith("W+"):
                    score = -1
                elif result == "Draw":
                    score = 0
                else:
                    raise ValueError(f"Invalid result: {result}")

                role = game.root.get("GN").split("_")[1]
                if role == "black":
                    self._add_result(player1, player2, score)
                elif role == "white":
                    self._add_result(player2, player1, score)
                else:
                    raise ValueError(f"Invalid role: {role}")

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
    elo.load_results('results/pit-puct')

    print("#players:", len(elo.players))
    print("#results:", len(elo.results))

    print("================================")

    # ratings = elo.elo_rating()
    ratings, model = elo.elo_rating_sgd()

    baseline_player, baseline_rating = "uct", 1000
    df = pd.DataFrame(ratings.items(), columns=['player', 'rating'])
    df.set_index('player', inplace=True)
    df['rating'] += baseline_rating - ratings[baseline_player]
    df.sort_values('rating', inplace=True, ascending=False)

    for player, rating in df.itertuples():
        print(f'{player}: {rating:.0f} ({elo.players[player].stats})')

    df.to_csv('results/puct_elo_rating.csv')

    df2 = df[df.index.str.startswith('puct')].copy()
    df2['epoch'] = df2.index.str.split('-').str[1].astype(int)
    df2.sort_values('epoch', inplace=True)

    plt.plot(df2['epoch'], df2['rating'], marker='o', label='puct')
    plt.axhline(y=df.loc['random', 'rating'], color='tab:blue', linestyle='--', label='random')
    plt.axhline(y=df.loc['uct', 'rating'], color='tab:orange', linestyle='--', label='uct')
    plt.xlabel('Train Iteration')
    plt.ylabel('Elo Rating')
    plt.title('Elo Rating of AlphaZero')
    plt.tight_layout()
    plt.legend()
    plt.savefig('results/puct_elo_rating.png')

    # while True:
    #     p1, p2 = (
    #         torch.tensor([elo.players[p].id], dtype=torch.long)
    #         for p in input().split()
    #     )
    #     out = model((p1, p2))
    #     out = F.softmax(out, dim=1).detach().numpy()[0]
    #     print(f'p1 win: {out[0]:.3f}, draw: {out[1]:.3f}, p2 win: {out[2]:.3f}')
