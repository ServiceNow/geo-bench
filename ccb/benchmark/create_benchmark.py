from math import ceil, floor
import random
from typing import Dict, List
from ccb import io
import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


def make_subsampler(max_sizes):
    def _subsample(partition, label_map, rng=np.random):
        return subsample(partition=partition, max_sizes=max_sizes, rng=rng)

    return _subsample


def subsample(partition: io.Partition, max_sizes: Dict[str, int], rng=np.random) -> io.Partition:

    new_partition = io.Partition()

    for split_name, sample_names in partition.partition_dict.items():
        if len(sample_names) > max_sizes[split_name]:
            subset = list(rng.choice(sample_names, max_sizes[split_name], replace=False))
        else:
            subset = sample_names[:]  # create a copy to avoid potential issues
        new_partition.partition_dict[split_name] = subset
    return new_partition


def _make_split_label_maps(label_map: Dict[int, List[str]], partition_dict: Dict[str, List[str]]):
    """Organize label map into 'train', 'valid' and 'test'."""
    split_label_maps = {}
    reverse_label_map = {}
    for label, sample_names in label_map.items():
        for sample_name in sample_names:
            reverse_label_map[sample_name] = label
    for split, sample_names in partition_dict.items():
        split_label_maps[split] = defaultdict(list)
        for sample_name in sample_names:
            label = reverse_label_map[sample_name]
            split_label_maps[split][label].append(sample_name)
    return split_label_maps


def _filter_for_min_size(split_label_maps, min_class_sizes: Dict[str, int]):
    new_split_label_maps = defaultdict(dict)
    for label in split_label_maps["train"].keys():

        ok = True
        for split, min_class_size in min_class_sizes.items():
            if len(split_label_maps[split].get(label, ())) < min_class_size:
                ok = False
        if ok:
            for split in ("train", "valid", "test"):
                new_split_label_maps[split][label] = split_label_maps[split][label][:]

    return new_split_label_maps


def assert_no_overlap(split_label_maps: Dict[str, Dict[int, List[str]]]):
    sample_set = set()
    total_count = 0
    for label_map in split_label_maps.values():
        for sample_names in label_map.values():
            sample_set.update(sample_names)
            total_count += len(sample_names)

    assert len(sample_set) == total_count


def make_resampler(max_sizes, min_class_sizes={"train": 10, "valid": 1, "test": 1}):
    def _resample(partition, label_map, rng=np.random):
        return resample(
            partition=partition, label_map=label_map, max_sizes=max_sizes, min_class_sizes=min_class_sizes, rng=rng
        )

    return _resample


def resample(
    partition: io.Partition,
    label_map: Dict[int, List[str]],
    max_sizes: Dict[str, int],
    min_class_sizes: Dict[str, int],
    verbose=True,
    rng=np.random,
) -> io.Partition:

    split_label_maps = _make_split_label_maps(label_map, partition_dict=partition.partition_dict)
    assert_no_overlap(split_label_maps)
    new_split_label_maps = _filter_for_min_size(split_label_maps, min_class_sizes)
    assert_no_overlap(new_split_label_maps)
    partition_dict = defaultdict(list)
    for split, max_size in max_sizes.items():
        label_map = new_split_label_maps[split]
        max_sample_per_class = floor(max_size / len(label_map))
        for label, sample_names in label_map.items():
            if len(sample_names) > max_sample_per_class:
                label_map[label] = rng.choice(sample_names, size=max_sample_per_class, replace=False)

            partition_dict[split].extend(label_map[label])

    for sample_names in partition_dict.values():
        rng.shuffle(sample_names)  # shuffle in place the mutable sequence

    if verbose:
        print("Class rebalancing:")
        for split, label_map in split_label_maps.items():
            print(f"{split}")
            for label, sample_names in label_map.items():
                new_sample_names = new_split_label_maps[split].get(label, ())
                print(f"  class {label} size: {len(sample_names)} -> {len(new_sample_names)}.")
        print()
    return io.Partition(partition_dict=partition_dict)


def transform_dataset(
    dataset_dir: Path,
    new_benchmark_dir: Path,
    partition_name: str,
    resampler=None,
    sample_converter=None,
    delete_existing=False,
    hdf5=True,
):

    dataset = io.Dataset(dataset_dir, partition_name=partition_name)
    task_specs = dataset.task_specs
    label_map = task_specs.label_map
    task_specs.benchmark_name = new_benchmark_dir.name
    new_dataset_dir = new_benchmark_dir / dataset_dir.name

    if new_dataset_dir.exists() and delete_existing:
        print(f"Deleting exising dataset {new_dataset_dir}.")
        shutil.rmtree(new_dataset_dir)

    new_dataset_dir.mkdir(parents=True, exist_ok=True)

    if resampler is not None:
        new_partition = resampler(
            partition=dataset.load_partition(partition_name),
            label_map=label_map,
        )
    else:
        new_partition = dataset.load_partition(partition_name)

    task_specs.save(new_dataset_dir, overwrite=True)

    for split_name, sample_names in new_partition.partition_dict.items():
        print(f"  Converting {len(sample_names)} samples from {split_name} split.")
        for sample_name in tqdm(sample_names):

            if sample_converter is None:
                if hdf5:
                    sample_name += ".hdf5"
                    shutil.copyfile(dataset_dir / sample_name, new_dataset_dir / sample_name)
                else:
                    shutil.copytree(dataset_dir / sample_name, new_dataset_dir / sample_name, dirs_exist_ok=True)
            else:
                raise NotImplementedError()

    new_partition.save(new_dataset_dir, "default")


def _make_benchmark(new_benchmark_name, specs, src_benchmark_name="converted"):

    for dataset_name, (resampler, sample_converter) in specs.items():
        print(f"Transforming {dataset_name}.")
        transform_dataset(
            dataset_dir=io.CCB_DIR / src_benchmark_name / dataset_name,
            new_benchmark_dir=io.CCB_DIR / new_benchmark_name,
            partition_name="default",
            resampler=resampler,
            sample_converter=sample_converter,
            delete_existing=True,
        )


def make_classification_benchmark():

    default_resampler = make_resampler(max_sizes={"train": 5000, "valid": 1000, "test": 1000})
    specs = {
        "eurosat": (default_resampler, None),
        "brick_kiln_v1.0": (default_resampler, None),
        # "so2sat": (default_resampler, None),
        "pv4ger_classification": (default_resampler, None),
        # "geolifeclef-2021": ({"train": 10000, "valid": 5000, "test": 5000}, None),
    }
    _make_benchmark("classification_v0.3", specs)


def make_segmentation_benchmark():

    default_resampler = make_subsampler(max_sizes={"train": 5000, "valid": 1000, "test": 1000})
    specs = {
        # "pv4ger_segmentation": (default_resampler, None),
        # "forestnet_v1.0": (default_resampler, None),
        "cvpr_chesapeake_landcover": (default_resampler, None),
    }
    _make_benchmark("segmentation_v0.1", specs)


if __name__ == "__main__":
    make_classification_benchmark()
    # make_segmentation_benchmark()
