from typing import Dict
from ccb import io
import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path


def subsample(partition: io.Partition, max_sizes: Dict[str, int], rng=np.random) -> io.Partition:

    new_partition = io.Partition()

    for split_name, sample_names in partition.items():
        if len(sample_names) > max_sizes[split_name]:
            new_partition[split_name] = rng.choice(sample_names, max_sizes[split_name], replace=False)
        else:
            new_partition[split_name] = sample_names[:]  # create a copy to avoid potential issues

    return new_partition


def transform_dataset(dataset_dir: Path, new_benchmark_dir, partition_name, max_sizes, sample_converter=None):
    dataset = io.Dataset(dataset_dir, partition_name=partition_name)
    task_specs = dataset.task_specs
    task_specs.benchmark_name = new_benchmark_dir.name
    new_dataset_dir = new_benchmark_dir / dataset_dir.name
    new_dataset_dir.mkdir(parents=True, exist_ok=True)

    new_partition = subsample(dataset.load_partition(partition_name), max_sizes)

    task_specs.save(new_dataset_dir, overwrite=True)

    for split_name, sample_names in new_partition.items():
        print(f"  Converting {len(sample_names)} from {split_name} split.")
        for sample_name in tqdm(sample_names):
            if sample_converter is None:
                shutil.copytree(dataset_dir / sample_name, new_dataset_dir / sample_name, dirs_exist_ok=True)
            else:
                raise NotImplementedError()

    new_partition.save(new_dataset_dir, "default")


def make_benchmark(new_benchmark_name, specs, src_benchmark_name="converted"):

    for dataset_name, (max_sizes, sample_converter) in specs.items():
        print(f"Transforming {dataset_name}.")
        transform_dataset(
            dataset_dir=io.CCB_DIR / src_benchmark_name / dataset_name,
            new_benchmark_dir=io.CCB_DIR / new_benchmark_name,
            partition_name="default",
            max_sizes=max_sizes,
            sample_converter=sample_converter,
        )


def make_classification_benchmark():

    sizes = {"train": 5000, "valid": 1000, "test": 1000}
    specs = {"eurosat": (sizes, None), "brick_kiln_v1.0": (sizes, None)}
    make_benchmark("classification", specs)


if __name__ == "__main__":
    make_classification_benchmark()
