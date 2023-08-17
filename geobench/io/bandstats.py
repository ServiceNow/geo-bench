"""Compute dataset band statistics for each band and save them in bandstats.json.

For the future, implement partitions and splits
"""
import argparse
import json
import os
from collections import OrderedDict, defaultdict
from typing import DefaultDict, Dict, List

import numpy as np
from tqdm import tqdm

from geobench import io
from geobench.io.dataset import compute_dataset_statistics

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="path to CCB dataset folder")
parser.add_argument("-s", "--splits", default=False, help="compute statistics separately for each split")
parser.add_argument("--values-per-image", default=1000, help="values per image")
parser.add_argument("--samples", default=1000, help="dataset subset size")


def produce_band_stats(dataset: io.GeobenchDataset, values_per_image: int = 1000, samples: int = 1000) -> None:
    """Compute and save band statistics.

    Args:
        dataset: GeobenchDataset
        values_per_image: number of values to consider per image
        sample: number of samples
    """
    # if use_splits:
    #     for partition in dataset.list_partitions():
    #         dataset.set_partition(partition)
    #         for split in dataset.list_splits():
    #             print(f"Computing statistics for {partition}:{split}")
    #             dataset.set_split(split)
    #             _band_values, band_stats = compute_dataset_statistics(
    #                 dataset, n_value_per_image=values_per_image, n_samples=samples
    #             )

    #             stats_fname = os.path.join(dataset.dataset_dir, f"{partition}_{split}_band_stats.json")
    #             with open(stats_fname, "w", encoding="utf8") as fp:
    #                 json.dump({k: v.to_dict() for k, v in band_stats.items()}, fp, indent=4, sort_keys=True)
    #             print("-> Dumped statistics to {}".format(stats_fname))
    # else:
    dataset.set_partition("default")
    dataset.set_split("train")
    print("Computing single statistics for whole dataset")
    _band_values, band_stats = compute_dataset_statistics(
        dataset, n_value_per_image=values_per_image, n_samples=samples
    )
    stats_fname = os.path.join(dataset.dataset_dir, "band_stats.json")
    with open(stats_fname, "w", encoding="utf8") as fp:
        json.dump({k: v.to_dict() for k, v in band_stats.items()}, fp, indent=4, sort_keys=True)
    print(f"Statistics written to {stats_fname}.")


def produce_all_band_stats(benchmark_dir: str) -> None:
    """Compute all band statistics for a benchmark.

    Args:
        benchmark_name: path to the benchmark directory
    """
    for task in io.task_iterator(benchmark_dir=benchmark_dir):
        print(f"Producing bandstats for dataset {task.dataset_name} of benchmark {os.path.basename(benchmark_dir)}.")
        produce_band_stats(task.get_dataset(benchmark_dir=benchmark_dir, split=None))


if __name__ == "__main__":
    produce_all_band_stats(benchmark_dir=io.GEO_BENCH_DIR / "segmentation_v0.2")
