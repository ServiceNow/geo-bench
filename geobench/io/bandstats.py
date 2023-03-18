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


# def linear_remapping(band_array):
#     """Map band array linearly to a certain range.

#     Args:
#         band_array: array of band data

#     Returns:
#         Linearlz remapped band data
#     """
#     lower_output = 0
#     upper_output = 1

#     lower_input = np.percentile(band_array, 1)
#     upper_input = np.percentile(band_array, 99)

#     output_band_array = (band_array - lower_input) * (
#         (upper_output - lower_output) / (upper_input - lower_input)
#     ) + lower_output

#     return output_band_array


# def own_normalization_stats(dataset):
#     """Check normalization stats computation."""
#     accumulator: DefaultDict[str, List] = defaultdict(list)

#     indices = np.random.choice(len(dataset), 1000, replace=False)

#     for i in tqdm(indices, desc="Extracting Statistics"):
#         sample = dataset[i]

#         for band in sample.bands:
#             accumulator[band.band_info.name].append(band.data.flatten())

#     band_stats = {}
#     remapped_band_stats = {}
#     for name, values in accumulator.items():
#         stacked_values = np.hstack(values)

#         remapped = linear_remapping(stacked_values)
#         band_stats[name] = {"mean": stacked_values.mean(), "std": stacked_values.std()}
#         remapped_band_stats[name] = {"mean": remapped.mean(), "std": remapped.std()}

#     # test normalization of bands
#     indices = np.random.choice(len(dataset), 10, replace=False)
#     for i in indices:
#         # current approach
#         sample = dataset[i]

#         for band in sample.bands:
#             band_name = band.band_info.name
#             if band_name in ["04 - Red", "03 - Green", "02 - Blue", "Red", "Green", "Blue"]:
#                 print("Current / alternative appraoch:")
#                 current = (band.data - band_stats[band_name]["mean"]) / band_stats[band_name]["std"]
#                 print(f"{band_name}, mean: {current.mean()}, std: {current.std()}")
#                 new = (linear_remapping(band.data) - remapped_band_stats[band_name]["mean"]) / remapped_band_stats[
#                     band_name
#                 ]["std"]
#                 print(f"{band_name}, mean: {new.mean()}, std: {new.std()}")

#     return band_stats


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
    produce_all_band_stats(benchmark_dir="/mnt/data/cc_benchmark/segmentation_v0.2")
