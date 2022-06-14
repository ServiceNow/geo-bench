"""
Compute dataset band statistics for each band and save them in bandstats.json

For the future, implement partitions and splits
"""
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from warnings import warn

import numpy as np
from tqdm import tqdm

from ccb import io
from ccb.io import bandstats
from ccb.io.task import TaskSpecifications


def load_label(sample_path):
    if sample_path.suffix == ".hdf5":
        sample = io.load_sample_hdf5(sample_path, label_only=True)
        label = sample.label
    else:
        label_file = Path(sample_path, "label.json")
        with open(label_file, "r") as fd:
            label = json.load(fd)
    return label


def clean_partition(partition: io.Partition):
    all_samples = set()
    squeezed_out = []
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


def get_samples_and_verify_partition(dataset_dir, partition_name="default", max_count=None):
    dataset = io.Dataset(dataset_dir)
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


def load_label_map(dataset_dir, max_count=None):

    sample_paths = get_samples_and_verify_partition(dataset_dir, max_count=max_count)

    label_map = defaultdict(list)

    for sample_path in tqdm(sample_paths, desc="Loading labels."):
        label = load_label(sample_path)
        label_map[label].append(sample_path.stem)
    return label_map


def load_label_stats(task: TaskSpecifications, max_count=None):
    dataset_dir = task.get_dataset_dir()
    sample_paths = get_samples_and_verify_partition(dataset_dir, max_count=max_count)

    # label_stats = np.zeros((len(sample_paths), task.label_type.n_classes))
    # sample_names = []
    label_stats = {}

    for sample_path in tqdm(sample_paths, desc="Loading labels."):
        label = load_label(sample_path)

        label_stats[sample_path.stem] = list(task.label_type.label_stats(label))

    return label_stats


def write_all_label_map(benchmark_name="converted", max_count=None, compute_band_stats=True, task_filter=None):
    for task in io.task_iterator(benchmark_name=benchmark_name):

        if task_filter is not None and task_filter(task):

            dataset_dir = task.get_dataset_dir()

            print(f"Working with {dataset_dir}.")
            if compute_band_stats:
                try:
                    print(f"Producing Band Stats for {task.dataset_name}.")
                    bandstats.produce_band_stats(task.get_dataset(split=None))
                except Exception as e:
                    print(e)

            if task.label_type.__class__.__name__ == "Classification":

                print(f"Producing Label Map for {task.dataset_name}.")
                label_map = load_label_map(dataset_dir, max_count=max_count)

                print_label_map(label_map)
                with open(dataset_dir / "label_map.json", "w") as fp:
                    json.dump(label_map, fp, indent=4, sort_keys=True)

            else:
                label_stats = load_label_stats(task, max_count=max_count)
                print_label_stats(label_stats)
                with open(dataset_dir / "label_stats.json", "w") as fp:
                    json.dump(label_stats, fp, indent=4, sort_keys=True)

        else:
            print(f"Skipping task {task.dataset_name}.")


def print_label_stats(label_stats: Dict[str, List]):
    label_stats_array = np.array(list(label_stats.values()))
    cum_per_label = np.sum(label_stats_array, axis=0)
    for i, count in enumerate(cum_per_label):
        print(f"class {i:2d}: {count}")


def print_label_map(label_map, prefix="  ", max_count=200):
    lenghts = [(key, len(values)) for key, values in label_map.items()]
    lenghts.sort(key=lambda items: items[1], reverse=True)

    for i, (key, _length) in enumerate(lenghts[:max_count]):
        print(f"{i:3d} {prefix}class {key}: {len(label_map[key])} samples.")


def view_label_map_count(benchmark_name="converted"):
    for task in io.task_iterator(benchmark_name=benchmark_name):

        label_map = task.label_map
        print(f"Label map for dataset {task.dataset_name} of type {task.label_type.__class__.__name__}.")
        if label_map is not None:
            print_label_map(label_map)
        else:
            print(f"  Missing label_map for dataset {task.dataset_name}.")
        print()


def task_filter(task: TaskSpecifications):
    return isinstance(task.label_type, io.SegmentationClasses)


if __name__ == "__main__":
    write_all_label_map(benchmark_name="converted", max_count=None, compute_band_stats=False, task_filter=task_filter)
    # view_label_map_count()
