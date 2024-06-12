from datetime import datetime

def format_datetime(dt: datetime | None = None) -> str:
    dt = dt or datetime.now()
    return dt.strftime('%Y%m%d-%H%M%S')


class GameData:
    PLAYER_MAP = {1: 'B', -1: 'W'}

    def __init__(self, n: int, **kwargs):
        self.n = n

        self.attr = {'GM': '1', 'FF': '4', 'CA': 'UTF-8', 'SZ': n}
        for key, value in kwargs.items():
            self.attr[key] = value

        self.moves: list[tuple[int, int]] = []

    def __getitem__(self, key: str):
        return self.attr[key]
    
    def __setitem__(self, key: str, value: object):
        self.attr[key] = value

    def add_move(self, player: int, action: int):
        self.moves.append((player, action))

    def set_result(self, result: float, score: float | None = None):
        if result == 1:
            self.attr['RE'] = f'B+{score if score is not None else "?"}'
        elif result == -1:
            self.attr['RE'] = f'W+{score if score is not None else "?"}'
        else:
            self.attr['RE'] = 'Draw'

    def to_sgf(self) -> str:
        sgf = ['', ''.join(f'{key}[{value}]' for key, value in self.attr.items())]
        for player, action in self.moves:
            tag = GameData.PLAYER_MAP[player]
            if action == self.n * self.n:
                pos = ''
            else:
                x, y = action % self.n, action // self.n
                pos = chr(ord('a') + x) + chr(ord('a') + y)
            sgf.append(f'{tag}[{pos}]')
        return '(' + ';'.join(sgf) + ')'
