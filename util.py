
PLAYER_MAP = {1: 'B', -1: 'W'}

def history_to_sgf(n: int, history: list[tuple[int, int]]):
    sgf = [f';GM[1]FF[4]CA[UTF-8]SZ[{n}]']
    for player, action in history:
        tag = PLAYER_MAP[player]
        if action == n * n:
            pos = ''
        else:
            x, y = action % n, action // n
            pos = chr(ord('a') + x) + chr(ord('a') + y)
        sgf.append(f';{tag}[{pos}]')
    return '(' + ''.join(sgf) + ')'
