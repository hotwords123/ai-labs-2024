from env.go.go_env import GoGame
from env.base_env import BLACK, WHITE, EMPTY, EPS
import numpy as np
import unittest


class TestGoGame(unittest.TestCase):
    def assertAllEqual(self, a, b):
        self.assertTrue(np.all(a == b), f"{a} != {b}")

    def test_init(self):
        game = GoGame(5)
        game.reset()

        self.assertEqual(game.n, 5)
        self.assertEqual(game.action_space_size, 26)
        self.assertEqual(game.current_player, BLACK)
        self.assertEqual(game.ended, False)
        self.assertAllEqual(game.observation, EMPTY)
        self.assertAllEqual(game.action_mask, True)

    def test_step(self):
        game = GoGame(5)
        game.reset()

        for action in [
            12,  6, 11,  7, 13,  8,  9, 10, 15, 19,
             5, 14,  1, 18, 17,  4,  2, 21, 22, 23,
            10, 16, 20, 21, 16,  9,  0, 25,
        ]:
            obs, reward, done = game.step(action)
            self.assertEqual(reward, 0)
            self.assertEqual(done, False)

        self.assertEqual(game.current_player, BLACK)

        obs, reward, done = game.step(25)
        self.assertAllEqual(obs, np.array([
            [ 1,  1,  1,  0, -1],
            [ 1, -1, -1, -1, -1],
            [ 1,  1,  1,  1, -1],
            [ 1,  1,  1, -1, -1],
            [ 1,  0,  1, -1,  0],
        ], dtype=np.int32))
        self.assertEqual(reward, 1)
        self.assertEqual(done, True)

    def test_reward(self):
        game = GoGame(5)
        game.reset()
        game.board.load_numpy(np.array([
            [ 0, -1,  1,  1,  1],
            [-1, -1,  1,  0,  0],
            [-1,  1, -1, -1, -1],
            [-1,  1,  1,  1, -1],
            [ 0,  1,  0,  1, -1],
        ], dtype=np.int32))
        self.assertEqual(game.current_player, BLACK)

        # 1. Black wins
        game1 = game.fork()
        game1.step(25)
        game1.step(0)
        game1.step(20)
        game1.step(25)
        obs, reward, done = game1.step(25)
        self.assertAllEqual(obs, np.array([
            [ 0,  0,  1,  1,  1],
            [ 0,  0,  1,  0,  0],
            [ 0,  1, -1, -1, -1],
            [ 0,  1,  1,  1, -1],
            [ 1,  1,  0,  1, -1],
        ], dtype=np.int32))
        self.assertEqual(reward, 1)  # Current player is BLACK, so reward is positive
        self.assertEqual(done, True)

        # 2. White wins
        game2 = game.fork()
        game2.step(20)
        game2.step(22)
        game2.step(25)
        obs, reward, done = game2.step(25)
        self.assertAllEqual(obs, np.array([
            [ 0, -1,  1,  1,  1],
            [-1, -1,  1,  0,  0],
            [-1,  0, -1, -1, -1],
            [-1,  0,  0,  0, -1],
            [ 0,  0, -1,  0, -1],
        ], dtype=np.int32))
        self.assertEqual(reward, 1)  # Current player is WHITE, so reward is positive
        self.assertEqual(done, True)

        # 3. Draw
        game.step(25)
        obs, reward, done = game.step(25)
        self.assertAllEqual(obs, np.array([
            [ 0, -1,  1,  1,  1],
            [-1, -1,  1,  0,  0],
            [-1,  1, -1, -1, -1],
            [-1,  1,  1,  1, -1],
            [ 0,  1,  0,  1, -1],
        ], dtype=np.int32))
        self.assertAlmostEqual(reward, 0, delta=EPS)
        self.assertEqual(done, True)


if __name__ == "__main__":
    unittest.main()
