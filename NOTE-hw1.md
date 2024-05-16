# Homework 1

## 进行实验

运行 `pit.py` 进行实验对局，命令行参数的使用方法可以通过 `python -m pit --help` 查看。

使用 `exp.sh` 并行运行多个实验，通过 `num_procs` 环境变量可以指定并行的进程数。每个实验对应的命令如下。

### 2.3 MCTS 算法与其它算法对弈

```bash
./exp.sh battle
```

### 2.4 搜索 $C$ 的最佳取值

```bash
./exp.sh search_c
```

### 2.5 比较 $C$ 的取值对输出策略的影响

```bash
./exp.sh compare_c
```

### 2.6 MCTS 算法四子棋对局

```bash
./exp.sh gobang_battle 100
```

## 数据分析

代码使用了 `scipy` 和 `pandas` 包进行数据分析，需要在运行 `plot.ipynb` 之前安装。

`plot.ipynb` 包含了对实验 2.4 作图的代码，以及实验 2.6 收集数据的代码。
