"""
Compute dataset band statistics for each band and save them in bandstats.json

For the future, implement partitions and splits
"""
import json
from pathlib import Path
from collections import defaultdict
from warnings import warn
from ccb import io
from tqdm import tqdm
import numpy as np


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


def get_samples_and_verify_partition(dataset_dir, partition_name="default"):
    dataset = io.Dataset(dataset_dir)
    partition = dataset.load_partition(partition_name)

    partition, all_samples, squeezed_out, size_difference = clean_partition(partition)
    if size_difference != 0:
        answer = input(
            f"The partition of {dataset_dir} had {size_difference} redundent elements.{', '.join(squeezed_out)}\n Would you like to overwrite it? y/n."
        )
        if answer.lower() == "y":
            partition.save(dataset_dir, partition_name)

    sample_names = []
    for file_name in tqdm(
        list(dataset_dir.glob("*")), desc=f"Collecting list of subdirectories in {dataset_dir.name}."
    ):
        if file_name.is_dir() or file_name.suffix == ".hdf5":
            sample_names.append(file_name)

    if len(all_samples) != len(sample_names):
        warn(
            f"Partition {partition_name} has {len(all_samples)}, but there is {len(sample_names)} samples in the directory."
        )
    return sample_names


def load_label_map(dataset_dir, max_count=None):

    sample_paths = get_samples_and_verify_partition(dataset_dir)

    label_map = defaultdict(list)

    if max_count is not None and len(sample_paths) > max_count:
        sample_paths = np.random.choice(sample_paths, max_count)

    for sample_path in tqdm(sample_paths, desc="Loading labels."):
        label = load_label(sample_path)
        label_map[label].append(sample_path.stem)
    return label_map


def write_all_label_map(benchmark_name="converted", max_count=None):
    for task in io.task_iterator(benchmark_name=benchmark_name):

        dataset_dir = task.get_dataset_dir()
        if task.label_type.__class__.__name__ != "Classification":
            print(f"Skipping {task.dataset_name}.")
            continue

        print(f"Producing Label Map for {dataset_dir}.")
        label_map = load_label_map(dataset_dir, max_count=max_count)

        print_label_map(label_map)

        with open(dataset_dir / "label_map.json", "w") as fp:
            json.dump(label_map, fp, indent=4, sort_keys=True)


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


if __name__ == "__main__":
    write_all_label_map(max_count=None)
    # view_label_map_count()
