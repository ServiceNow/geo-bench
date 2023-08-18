"""Compute dataset band statistics for each band and save them in bandstats.json.

For the future, implement partitions and splits
"""
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set
from warnings import warn

import numpy as np
from tqdm import tqdm

from geobench import io
from geobench.io import bandstats
from geobench.io.task import TaskSpecifications
from geobench import config

def load_label(sample_path):
    """Load a label.

    Args:
        sample_path: path to the sample

    Returns:
        loaded sample
    """
    if sample_path.suffix == ".hdf5":
        sample = io.load_sample_hdf5(sample_path, label_only=True)
        label = sample.label
    else:
        label_file = Path(sample_path, "label.json")
        with open(label_file, "r") as fd:
            label = json.load(fd)
    return label


def clean_partition(partition: io.Partition):
    """Clean partition.

    Args:
        partition: partition to clean
    """
    all_samples: Set[str] = set()
    squeezed_out: List[str] = []
    original_count = 0
    for split in ("train", "valid", "test"):
        samples = partition.partition_dict[split]
        original_count += len(samples)
        intersect = all_samples.intersection(samples)
        squeezed_out.extend(intersect)
        cleaned_samples = list(set(samples).difference(intersect))
        partition.partition_dict[split] = cleaned_samples
        all_samples.update(samples)

    return partition, all_samples, squeezed_out, original_count - len(all_samples)


def get_samples_and_verify_partition(dataset_dir, partition_name="default", max_count=None) -> List[str]:
    """Retrieve samples and verify partition.

    Args:
        dataset_dir: path to dataset directory
        partition_name: name of partition
        max_count: max count

    Returns:
        list of samples
    """
    dataset = io.GeobenchDataset(dataset_dir)
    partition = dataset.load_partition(partition_name)

    partition, all_samples, squeezed_out, size_difference = clean_partition(partition)
    if size_difference != 0:
        answer = input(
            f"The partition of {dataset_dir} had {size_difference} redundent elements.\n Would you like to overwrite it? y/n."
        )
        if answer.lower() == "y":
            partition.save(dataset_dir, partition_name)

    sample_names = []
    paths = list(dataset_dir.glob("*"))
    if max_count is not None:
        paths = np.random.choice(paths, max_count, replace=False)

    for file_name in tqdm(paths, desc=f"Collecting list of subdirectories in {dataset_dir.name}."):
        if file_name.is_dir() or file_name.suffix == ".hdf5":
            sample_names.append(file_name)

    if len(all_samples) != len(sample_names):
        warn(
            f"Partition {partition_name} has {len(all_samples)}, but there is {len(sample_names)} samples in the directory."
        )
    return sample_names


def load_label_map(dataset_dir: str, max_count: int = None) -> Dict[str, List[str]]:
    """Load label map, which maps labels to sample names.

    Args:
        dataset_dir: path to dataset directory
        max_count: max count

    Return:
        label map
    """
    sample_paths = get_samples_and_verify_partition(dataset_dir, max_count=max_count)

    label_map = defaultdict(list)

    for sample_path in tqdm(sample_paths, desc="Loading labels."):
        label = load_label(sample_path)
        label_map[label].append(sample_path.stem)
    return label_map


def load_label_stats(task_specs: TaskSpecifications, benchmark_dir: str, max_count: int = None):
    """Load label statistics.

    Args:
        task_specs: task specifications
        benchmark_dir: path to benchmark directory to retrieve datasets
        max_count: max count

    Returns:
        label statistics
    """
    dataset_dir = task_specs.get_dataset_dir(benchmark_dir=benchmark_dir)
    sample_paths = get_samples_and_verify_partition(dataset_dir, max_count=max_count)

    # label_stats = np.zeros((len(sample_paths), task_specs.label_type.n_classes))
    # sample_names = []
    label_stats = {}

    for sample_path in tqdm(sample_paths, desc="Loading labels."):
        label = load_label(sample_path)

        label_stats[sample_path.stem] = task_specs.label_type.label_stats(label).tolist()

    return label_stats


def write_all_label_map(
    benchmark_name: str = "converted", max_count: int = None, compute_band_stats: bool = True, task_filter=None
) -> None:
    """Write all label maps for a benchmark.

    Args:
        benchmark_name: name of benchmark
        max_count: max count
        compute_band_stats: whether or not to compute band statistics
        task_filter: filter out some tasks
    """
    benchmark_dir = str(io.GEO_BENCH_DIR / benchmark_name / "geobench")
    for task in io.task.task_iterator(benchmark_dir=benchmark_dir):

        if task_filter is not None and not task_filter(task):

            dataset_dir = task.get_dataset_dir(benchmark_dir=benchmark_dir)

            print(f"Working with {dataset_dir}.")
            if compute_band_stats:
                try:
                    print(f"Producing Band Stats for {task.dataset_name}.")
                    bandstats.produce_band_stats(task.get_dataset(benchmark_dir=benchmark_dir, split=None))
                except Exception as e:
                    print(e)

            if task.label_type.__class__.__name__ == "Classification":

                print(f"Producing Label Map for {task.dataset_name}.")
                label_map = load_label_map(dataset_dir, max_count=max_count)

                print_label_map(label_map)
                with open(dataset_dir / "label_map.json", "w") as fp:
                    json.dump(label_map, fp, indent=4, sort_keys=True)

            else:
                label_stats = load_label_stats(task, benchmark_dir=benchmark_dir, max_count=max_count)
                print_label_stats(label_stats)
                with open(dataset_dir / "label_stats.json", "w") as fp:
                    json.dump(label_stats, fp, indent=4, sort_keys=True)

        else:
            print(f"Skipping task {task.dataset_name}.")


def print_label_stats(label_stats: Dict[str, List]) -> None:
    """Print label statistics.

    Args:
        label_stats: label statistics
    """
    label_stats_array = np.array(list(label_stats.values()))
    cum_per_label = np.sum(label_stats_array, axis=0)
    for i, count in enumerate(cum_per_label):
        print(f"class {i:2d}: {count}")


def print_label_map(label_map, prefix: str = "  ", max_count: int = 200) -> None:
    """Print label mapping.

    Args:
        label_map: label mapping
        prefix: prefix to add
        max_count: max count
    """
    lenghts = [(key, len(values)) for key, values in label_map.items()]
    lenghts.sort(key=lambda items: items[1], reverse=True)

    for i, (key, _length) in enumerate(lenghts[:max_count]):
        print(f"{i:3d} {prefix}class {key}: {len(label_map[key])} samples.")


def view_label_map_count(benchmark_dir: str = config.GEO_BENCH_DIR / "converted") -> None:
    """Print counts of label maps.

    Args:
        benchmark_name: name of benchmark
    """
    for task in io.task_iterator(benchmark_dir=benchmark_dir):

        label_map = task.get_label_map
        print(f"Label map for dataset {task.dataset_name} of type {task.label_type.__class__.__name__}.")
        if label_map is not None:
            print_label_map(label_map)
        else:
            print(f"  Missing label_map for dataset {task.dataset_name}.")
        print()


def task_filter(task: TaskSpecifications):
    """Filter tasks from a benchmark.

    Args:
        task: task specifications
    """
    return task.dataset_name.startswith("geolife")
    # return isinstance(task.label_type, io.SegmentationClasses)


if __name__ == "__main__":
    write_all_label_map(benchmark_name="converted", max_count=None, compute_band_stats=True, task_filter=task_filter)
    # view_label_map_count()
