from collections import defaultdict
from email.policy import default
from typing import Dict, List

import numpy as np

from ccb import io
from ccb.benchmark.create_benchmark import resample, resample_from_stats


def make_rand_partition(n=1000):
    sample_names = [f"{i:04}" for i in range(n)]
    splits = np.random.choice(["train", "valid", "test"], size=n, p=[0.8, 0.1, 0.1], replace=True)
    class_probs = [0.7, 0.2, 0.02, 0.08]
    labels = np.random.choice(list(range(len(class_probs))), size=n, p=class_probs, replace=True)
    partition_dict = defaultdict(list)
    label_map = defaultdict(list)
    reverse_label_map = {}

    eye = np.eye(len(class_probs))

    label_stats = {}
    for split, sample_name, label in zip(splits, sample_names, labels):
        partition_dict[split].append(sample_name)
        label_map[label].append(sample_name)
        reverse_label_map[sample_name] = label
        label_stats[sample_name] = eye[label]  # converts to one hot

    assert_no_verlap(partition_dict)
    return io.Partition(partition_dict=partition_dict), label_map, reverse_label_map, label_stats


def assert_no_verlap(partition_dict: Dict[str, List[str]]):
    sample_set = set()
    total_count = 0
    for sample_names in partition_dict.values():

        sample_set.update(sample_names)
        total_count += len(sample_names)

    assert total_count == len(sample_set), f"{total_count} != {len(sample_set)}"


def test_resample():
    partition, label_map, reverse_label_map, _ = make_rand_partition(n=1000)
    max_sizes = {"train": 100, "valid": 20, "test": 25}
    min_class_sizes = {"train": 10, "valid": 1, "test": 2}
    partition = resample(
        partition,
        label_map,
        max_sizes=max_sizes,
        min_class_sizes=min_class_sizes,
    )
    verify_partition(partition, max_sizes, min_class_sizes, reverse_label_map)


def verify_partition(partition, max_sizes, min_class_sizes=None, reverse_label_map=None):
    partition_dict = partition.partition_dict
    assert_no_verlap(partition_dict)
    split_label_map = {}
    for split, max_size in max_sizes.items():
        split_label_map[split] = defaultdict(list)
        sample_names = partition_dict[split]

        assert len(sample_names) <= max_size

        for sample_name in sample_names:
            label = reverse_label_map[sample_name]
            split_label_map[split][label].append(sample_name)

    for split, min_class_size in min_class_sizes.items():
        for label, sample_names in split_label_map[split].items():
            assert len(sample_names) >= min_class_size

    for split, label_map in split_label_map.items():
        print(split)
        for label, sample_names in label_map.items():
            print(f"  class {label:2d}: {len(sample_names)}.")


def test_resample_from_stats():
    partition, _, reverse_label_map, label_stats = make_rand_partition(n=10000)
    max_sizes = {"train": 100, "valid": 20, "test": 25}
    min_class_sizes = {"train": 10, "valid": 1, "test": 1}

    new_partition, prob_dict = resample_from_stats(
        partition, label_stats=label_stats, max_sizes=max_sizes, return_prob=True
    )

    verify_partition(new_partition, max_sizes, min_class_sizes, reverse_label_map)


if __name__ == "__main__":
    test_resample()
    # test_resample_from_stats()
