from datetime import datetime
from sgf import *

def format_datetime(dt: datetime | None = None) -> str:
    dt = dt or datetime.now()
    return dt.strftime('%Y%m%d-%H%M%S')


class PlayerStats:
    def __init__(self, n_win: int = 0, n_lose: int = 0, n_draw: int = 0):
        self.n_win = n_win
        self.n_lose = n_lose
        self.n_draw = n_draw

    def __repr__(self):
        return f"Win: {self.n_win}, Lose: {self.n_lose}, Draw: {self.n_draw}"

    @property
    def n_match(self):
        return self.n_win + self.n_lose + self.n_draw

    @property
    def win_rate(self):
        return self.n_win / self.n_match

    @property
    def unbeaten_rate(self):
        return 1 - self.n_lose / self.n_match

    def update(self, reward):
        if reward == 1:
            self.n_win += 1
        elif reward == -1:
            self.n_lose += 1
        else:
            self.n_draw += 1

    def __add__(self, other: "PlayerStats"):
        return PlayerStats(
            self.n_win + other.n_win,
            self.n_lose + other.n_lose,
            self.n_draw + other.n_draw
        )

    def __iadd__(self, other: "PlayerStats"):
        self.n_win += other.n_win
        self.n_lose += other.n_lose
        self.n_draw += other.n_draw
        return self

    def inverse(self):
        return PlayerStats(self.n_lose, self.n_win, self.n_draw)


class GameData:
    PLAYER_MAP = {1: "B", -1: "W"}

    def __init__(self, n: int, game: SGFGame | None = None, **kwargs):
        self.n = n
        self.game = game if game is not None else SGFGame()
        self.root = self.game.root
        self.tree = self.game.tree

        if game is None:
            props =  {"GM": 1, "FF": 4, "CA": "UTF-8", "SZ": n, **kwargs}
            for k, v in props.items():
                self.game.root.set(k, str(v))

    def __getitem__(self, key: str):
        return self.root.get(key)

    def __setitem__(self, key: str, value: object):
        self.root.set(key, str(value))

    def format_action(self, action: int) -> str:
        if action == self.n * self.n:
            return ""
        x, y = action % self.n, action // self.n
        return f"{chr(ord('a') + x)}{chr(ord('a') + y)}"

    def add_move(self, player: int, action: int, comment: str = ""):
        node = SGFNode()
        node.set(self.PLAYER_MAP[player], self.format_action(action))
        if comment:
            node.set("C", comment)
        self.tree.nodes.append(node)

    def set_result(self, result: float, score: float | None = None):
        if result == 1:
            self["RE"] = f"B+{score if score is not None else '?'}"
        elif result == -1:
            self["RE"] = f'W+{score if score is not None else "?"}'
        else:
            self["RE"] = "Draw"

    def to_sgf(self) -> str:
        return self.game.to_sgf()

    @classmethod
    def from_sgf(cls, text: str) -> "GameData":
        parser = SGFParser(text)
        collection = parser.parse()
        assert len(collection.games) == 1, "Only one game is supported"

        game = collection.games[0]
        return cls(int(game.root.get("SZ")), game=game)
