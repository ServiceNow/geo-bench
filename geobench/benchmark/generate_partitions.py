"""Script to generate growing train size partitions."""

from typing import List

import numpy as np

from geobench import io


def generate_train_size_sweep(
    partition: io.Partition, train_fractions: List[float], dataset_dir=None, rng=np.random, verbose=True
):
    """Generate growing train size partition for the dataset specified by `dataset_dir`."""
    train_set = partition.partition_dict["train"][:]
    for fraction in train_fractions:
        new_size = int(len(train_set) * fraction)
        new_train_set = rng.choice(train_set, size=new_size, replace=False)

        partition.partition_dict["train"] = list(new_train_set)
        partition_name = f"{fraction:.2f}x_train"
        if verbose:
            print(f"    Saving partition {partition_name}")
        partition.save(dataset_dir, partition_name=partition_name)


def generate_partitions_for_benchmark(benchmark_name, train_fractions):
    """Generate growing train size partition for all dataset in `benchmark_name`."""
    for task in io.task_iterator(benchmark_dir=io.CCB_DIR / benchmark_name):

        # if not task.dataset_name.startswith("forest"): continue

        print(f"Working with task: {task.dataset_name}.")
        dataset = task.get_dataset(split=None)
        print(f"  Using partition {dataset.active_partition_name} in directory {dataset.dataset_dir}.")

        partition = dataset.active_partition

        generate_train_size_sweep(
            partition=partition,
            train_fractions=train_fractions,
            dataset_dir=dataset.dataset_dir,
        )


if __name__ == "__main__":
    generate_partitions_for_benchmark("classification_v0.5", train_fractions=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
