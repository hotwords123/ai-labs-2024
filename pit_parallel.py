import subprocess
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch


@dataclass
class TaskParams:
    name: str
    args: list[str]


class PitParallel:
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.task_queue = queue.Queue()
        self.tasks: list[TaskParams] = []
        self.threads: list[threading.Thread] = []

    def add_task(self, *args, **kwargs):
        self.tasks.append(TaskParams(*args, **kwargs))

    def run(self, gpu_ids: list[int]):
        np.random.shuffle(self.tasks)
        for task in self.tasks:
            self.task_queue.put(task)

        for gpu_id in gpu_ids:
            thread = threading.Thread(target=self.worker, args=(gpu_id,))
            thread.start()
            self.threads.append(thread)

    def join(self):
        self.task_queue.join()

        for thread in self.threads:
            thread.join()

    def worker(self, gpu_id: int):
        while True:
            try:
                task: TaskParams = self.task_queue.get_nowait()
            except queue.Empty:
                break

            task_dir = self.out_dir / task.name
            task_dir.mkdir(parents=True, exist_ok=True)

            if (task_dir / "stdout.log").exists():
                print(f"Task {task.name} already done, skip")
                self.task_queue.task_done()
                continue

            args = [
                "python", "pit_puct_mcts.py",
                *task.args,
                "--sgf_path", str(task_dir / f"match.sgf"),
                "--device", f"cuda:{gpu_id}",
            ]

            print(f"Task {task.name} started on GPU {gpu_id}")
            print(args)

            result = subprocess.run(args, check=True, capture_output=True)

            print(f"Task {task.name} finished on GPU {gpu_id}")

            (task_dir / "stdout.log").write_bytes(result.stdout)
            (task_dir / "stderr.log").write_bytes(result.stderr)

            self.task_queue.task_done()


def main():
    pit = PitParallel("results/pit-puct")

    model_dir = Path("checkpoint/helium")
    models = [(f"puct-{i}", model_dir / f"train-{i}.pth.tar") for i in range(10, 201, 10)]

    common_args = [
        "--game", "go",
        "--args", "9",
        "--n_match", "20",
        "--n_search", "200",
        "--n_rollout", "12",
        "--temperature", "0.1",
        "--C", "1.0",
    ]

    # Random vs PUCT
    for name, model_path in models:
        task_name = f"random_vs_{name}"
        task_args = [
            *common_args,
            "--players", "random", "puct",
            "--model_path", str(model_path),
        ]
        pit.add_task(task_name, task_args)

    # PUCT vs PUCT
    for i in range(len(models) - 1):
        for j in range(max(0, i - 5), i):
            name1, model1 = models[i]
            name2, model2 = models[j]
            task_name = f"{name1}_vs_{name2}"
            task_args = [
                *common_args,
                "--players", "puct", "puct2",
                "--model_path", str(model1),
                "--model_path2", str(model2),
            ]
            pit.add_task(task_name, task_args)

    print(f"Total tasks: {len(pit.tasks)}")

    gpu_ids = list(range(torch.cuda.device_count()))
    pit.run(gpu_ids)
    pit.join()


if __name__ == "__main__":
    main()