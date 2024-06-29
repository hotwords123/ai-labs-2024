# AlphaZero 使用说明

## 自我对弈训练

运行 `alphazero.py` 或 `alphazero_parallel.py` 可以进行 AlphaZero 的自我对弈训练。

### 通用设置

- `--seed`：随机种子，默认值为 0。
- `--job_id`：任务 ID，默认值为 `{mode}_{datetime}_{seed}`。
- `--log_file`：日志保存路径，默认值为 `log.txt`。
- `--checkpoint_path`：模型保存路径，默认值为 `checkpoint`。
- `--sgf_path`：SGF 棋谱保存路径，默认值为 `sgf`。
- `--result_path`：辅助结果保存路径，默认值为 `results`。

### 环境设置

- `--game`：环境类型，默认值为 `go`。
- `--args`：环境参数，默认值为 `[9]`。

### MCTS 设置

- `--n_search`：MCTS 模拟次数，默认值为 200。
- `--temperature`：MCTS 温度参数，默认值为 1.0。
- `--eval_temperature`：评估时的温度参数，默认值为 0.1。
- `--C`：MCTS 探索常数，默认值为 1.0。

### 训练模式

训练模式通过 `train` 子命令启动，支持以下参数：
- `--last_epoch`：上次训练的迭代次数，默认值为 0。

#### 强化学习设置

- `--n_train_iter`：训练迭代轮数，默认值为 50。
- `--n_match_train`：每轮迭代自我对弈的训练对局数，默认值为 20。
- `--n_match_update`：判断模型更新的评估对局数，默认值为 20。
- `--max_queue_length`：训练样本队列的最大长度，默认值为 300000。
- `--update_threshold`：分数大于阈值时更新模型，默认值为 0.551。
- `--use_latest`：不进行更新评估，总是使用最新模型进行自我对弈。
- `--eval_every`：每隔多少轮迭代评估一次模型，默认值为 1。
- `--enable_resign`：在自我对弈中允许认输。
- `--resign_threshold`：认输的局面价值阈值，默认值为 -0.90。
- `--n_resign_min_turn`：认输前的最少回合数，默认值为 20。
- `--n_resign_low_turn`：若得分低于阈值连续多少回合则认输，默认值为 3。
- `--resign_test_ratio`：用于检验认输误判率的对局比例，默认值为 0.1。
- `--opening_moves`：开局阶段的步数，默认值为 10。

#### MCTS 相关设置

- `--with_noise`：在训练对局中使用 Dirichlet 噪声。
- `--dir_epsilon`：Dirichlet 噪声的 $\varepsilon$ 值，默认值为 0.25。
- `--dir_alpha`：Dirichlet 噪声的 $\alpha$ 值，默认值为 0.15。

#### 模型训练设置

- `--epochs`：模型训练的轮数，默认值为 10。
- `--batch_size`：模型训练的批次大小，默认值为 256。
- `--lr`：模型训练的学习率，默认值为 0.01。
- `--weight_decay`：模型训练的权重衰减，默认值为 1e-4。

### 评估模式

评估模式通过 `eval` 子命令启动，支持以下参数：

- `--n_match_eval`：与基准玩家对弈的评估对局数，默认值为 20。
- `--checkpoint-name`：评估时使用的模型检查点，默认值为 `best`。

建议使用下面的并行对弈测试以获得更全面的评估结果。

## 对弈测试

运行 `pit_puct_mcts.py` 可以进行模型与基准玩家以及模型之间的对弈测试，调用方法可以参考代码中的说明，以及 `pit_parallel.py` 的调用方式。

### 并行对弈测试

运行 `pit_parallel.py` 可以进行并行对弈测试。

- 使用 `--model_dir` 指定模型文件夹，例如 `checkpoint/{job_id}`。
- 使用 `--out_dir` 指定结果保存路径，默认为 `results/pit-puct`。
- 测试时调用 `pit_puct_mcts.py` 的参数由代码中的 `common_args` 指定。
- 其它测试配置可以在代码中修改。

## 结果分析

实验结果中损失函数曲线、胜率曲线和 Elo 等级分曲线的绘制代码位于 `plot_puct.ipynb`。
