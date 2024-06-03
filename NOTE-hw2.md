# Homework 2

## 进行测试

运行 `python -m test_examples.<test_name>` 进行测试。

## 和 AI 对弈

示例：在 5x5 的围棋棋盘上，执白棋与 MCTS 算法对弈。
```sh
python -m pit --game go --args 5 --players uct human --C 1.0 --n_rollout 8 --n_search 600 --deterministic
```
