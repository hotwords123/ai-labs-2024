from datetime import datetime
from typing import NamedTuple
import re

def format_datetime(dt: datetime | None = None) -> str:
    dt = dt or datetime.now()
    return dt.strftime('%Y%m%d-%H%M%S')


class GameMove(NamedTuple):
    player: int
    action: int
    comment: str


class GameData:
    PLAYER_MAP = {1: 'B', -1: 'W'}
    ESCAPE_CHARS = re.compile(r'[\]:\\]')

    def __init__(self, n: int, **kwargs):
        self.n = n

        self.attr = {'GM': '1', 'FF': '4', 'CA': 'UTF-8', 'SZ': n}
        for key, value in kwargs.items():
            self.attr[key] = value

        self.moves: list[GameMove] = []

    def __getitem__(self, key: str):
        return self.attr[key]
    
    def __setitem__(self, key: str, value: object):
        self.attr[key] = value

    def add_move(self, player: int, action: int, comment: str = ''):
        self.moves.append(GameMove(player, action, comment))

    def set_result(self, result: float, score: float | None = None):
        if result == 1:
            self.attr['RE'] = f'B+{score if score is not None else "?"}'
        elif result == -1:
            self.attr['RE'] = f'W+{score if score is not None else "?"}'
        else:
            self.attr['RE'] = 'Draw'

    def to_sgf(self) -> str:
        sgf = [''.join(f'{key}[{value}]' for key, value in self.attr.items())]
        for player, action, comment in self.moves:
            tag = GameData.PLAYER_MAP[player]
            if action == self.n * self.n:
                tag += '[]'
            else:
                x, y = action % self.n, action // self.n
                tag += f'[{chr(ord("a") + x)}{chr(ord("a") + y)}]'
            if comment:
                tag += f'C[{self.escape_text(comment)}]'
            sgf.append(tag)
        return '(;' + ';'.join(sgf) + ')'

    @staticmethod
    def escape_text(text: str) -> str:
        return GameData.ESCAPE_CHARS.sub(lambda c: f'\\{c.group()}', text)
