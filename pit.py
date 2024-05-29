from env import *
from players import *
from mcts.uct_mcts import UCTMCTSConfig
from tqdm import trange, tqdm

import numpy as np

def print_devide_line(n=50):
    print("--"*n)

def pit(game:BaseGame, player1:BasePlayer, player2:BasePlayer, log_output:bool=True):
    game.reset()
    if isinstance(player1, UCTPlayer):
        player1.mcts = None
    if isinstance(player2, UCTPlayer):
        player2.mcts = None
    if log_output:
        print(f"start playing {type(game)}")
        print_devide_line()
    reward = 0
    
    for player in [player1, player2]:
        if player.__class__.__name__ == 'UCTPlayer':
            player.clear()
            
    while True:
        a1 = player1.play(game)
        _, reward, done = game.step(a1)
        if player2.__class__.__name__ == 'UCTPlayer':
            player2.opp_play(a1)
        if log_output:
            print(f"Player 1 ({player1}) move: {a1}")
            print(game.to_string())
            print_devide_line()
        if done:
            break
        a2 = player2.play(game)
        _, reward, done = game.step(a2)
        if player1.__class__.__name__ == 'UCTPlayer':
            player1.opp_play(a2)
        if log_output:
            print(f"Player 2 ({player2}) move: {a2}")
            print(game.to_string())
            print_devide_line()
        if done:
            reward *= -1
            break
    if log_output:
        if reward == 1:
            print(f"Player 1 ({player1}) win")
        elif reward == -1:
            print(f"Player 2 ({player2}) win")
        else:
            print("Draw")
    return reward
        
def multi_match(game:BaseGame, player1:BasePlayer, player2:BasePlayer, n_match=100):
    n_p1_win, n_p2_win, n_draw = 0, 0, 0
    T = trange(n_match)
    for _ in T:
        reward = pit(game, player1, player2, log_output=False)
        if reward == 1:
            n_p1_win += 1
        elif reward == -1:
            n_p2_win += 1
        else:
            n_draw += 1
        T.set_description_str(f"P1 win: {n_p1_win} ({n_p1_win}) P2 win: {n_p2_win} ({n_p2_win}) Draw: {n_draw} ({n_draw})")        
    print(f"Player 1 ({player1}) win: {n_p1_win} ({n_p1_win/n_match*100:.2f}%)")
    print(f"Player 2 ({player2}) win: {n_p2_win} ({n_p2_win/n_match*100:.2f}%)")
    print(f"Draw: {n_draw} ({n_draw/n_match*100:.2f}%)")
    print(f"Player 1 not lose: {n_p1_win+n_draw} ({(n_p1_win+n_draw)/n_match*100:.2f}%)")
    print(f"Player 2 not lose: {n_p2_win+n_draw} ({(n_p2_win+n_draw)/n_match*100:.2f}%)")
    return n_p1_win, n_p2_win, n_draw
        
        
def search_best_C():
    from matplotlib import pyplot as plt
    p2nl = []
    cs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0]
    n_match = 100
    for c in cs:
        config = UCTMCTSConfig()
        config.C = c
        config.n_rollout = 7
        config.n_search = 64
        player1 = AlphaBetaPlayer()
        player2 = UCTPlayer(config, deterministic=True)
        game = TicTacToeGame()
        p1w, p2w, drw = multi_match(game, player1, player2, n_match=n_match)
        p2nl.append((p2w+drw)/n_match)
    plt.plot(cs, p2nl)
    plt.savefig('p2nl.png')
        
if __name__ == '__main__':
    #####################
    # Modify code below #
    #####################
    import argparse

    GAME_CLASS = {
        'tictactoe': TicTacToeGame,
        'gobang': GobangGame,
        'go': GoGame,
    }

    PLAYER_CLASS = {
        'human': HumanPlayer,
        'random': RandomPlayer,
        'alphabeta': AlphaBetaPlayer,
        'uct': lambda: UCTPlayer(config, deterministic=args.deterministic, log_policy=args.log_policy),
        'uct2': lambda: UCTPlayer(config2, deterministic=args.deterministic, log_policy=args.log_policy),
    }

    parser = argparse.ArgumentParser(description='Pit two players against each other')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Do not print the game state')
    parser.add_argument('--game', choices=GAME_CLASS.keys(), default='tictactoe', help='Game to play')
    parser.add_argument('--args', type=int, nargs='*', help='Arguments for the game')
    parser.add_argument('--players', choices=PLAYER_CLASS.keys(), nargs=2, default=['random', 'alphabeta'], help='Players to play')
    parser.add_argument('--n_match', type=int, help='Number of matches to play')
    parser.add_argument('--C', type=float, default=1.0, help='C value for UCTPlayer')
    parser.add_argument('--n_rollout', type=int, default=7, help='Number of rollouts for UCTPlayer')
    parser.add_argument('--n_search', type=int, default=64, help='Number of searches for UCTPlayer')
    parser.add_argument('--deterministic', action='store_true', help='Deterministic UCTPlayer')
    parser.add_argument("--log_policy", action="store_true", help="Log policy of UCTPlayer")
    parser.add_argument('--C2', type=float, default=1.0, help='C value for UCTPlayer 2')
    parser.add_argument('--n_rollout2', type=int, default=7, help='Number of rollouts for UCTPlayer 2')
    parser.add_argument('--n_search2', type=int, default=64, help='Number of searches for UCTPlayer 2')

    args = parser.parse_args()

    # set seed to reproduce the result
    if args.seed is not None:
        np.random.seed(args.seed)
        
    game_args = args.args if args.args is not None else []
    game = GAME_CLASS[args.game](*game_args)
    
    # config for MCTS
    config = UCTMCTSConfig(
        C=args.C,
        n_rollout=args.n_rollout,
        n_search=args.n_search
    )

    config2 = UCTMCTSConfig(
        C=args.C2 or args.C,
        n_rollout=args.n_rollout2 or args.n_rollout,
        n_search=args.n_search2 or args.n_search
    )
    
    # player initialization    
    player1 = PLAYER_CLASS[args.players[0]]()
    player2 = PLAYER_CLASS[args.players[1]]()

    result_text = [
        f"Player 1 ({player1}) win",
        f"Player 2 ({player2}) win",
        "Draw"
    ]
    
    # single match
    if args.n_match is None:
        reward = pit(game, player1, player2, log_output=not args.quiet)
        if args.quiet:
            print(result_text[0 if reward > 0 else 1 if reward < 0 else 2])
    else:
        n_p1_win, n_p2_win, n_draw = multi_match(game, player1, player2, n_match=args.n_match)
        # for res, n_win in zip(result_text, [n_p1_win, n_p2_win, n_draw]):
        #     rate = n_win / args.n_match * 100
        #     print(f"{res}: {n_win} ({rate:.2f}%)")
    
    #####################