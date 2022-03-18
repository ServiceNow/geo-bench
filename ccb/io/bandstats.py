"""
Compute dataset band statistics for each band and save them in bandstats.json

For the future, implement partitions and splits
"""
import argparse
import json
import os

from ccb.io.dataset import Dataset, compute_dataset_statistics
from ccb import io


parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="path to CCB dataset folder")
parser.add_argument("-s", "--splits", default=False, help="compute statistics separately for each split")
parser.add_argument("--values-per-image", default=1000, help="values per image")
parser.add_argument("--samples", default=1000, help="dataset subset size")


def produce_band_stats(dataset_dir, use_splits=False, values_per_image=1000, samples=1000):

    dataset = Dataset(dataset_dir)

    if use_splits:
        for partition in dataset.list_partitions():
            dataset.set_partition(partition)
            for split in dataset.list_splits():
                print(f"Computing statistics for {partition}:{split}")
                dataset.set_split(split)
                _band_values, band_stats = compute_dataset_statistics(
                    dataset, n_value_per_image=values_per_image, n_samples=samples
                )
                stats_fname = os.path.join(dataset.dataset_dir, f"{partition}_{split}_band_stats.json")
                with open(stats_fname, "w", encoding="utf8") as fp:
                    json.dump({k: v.to_dict() for k, v in band_stats.items()}, fp, indent=4, sort_keys=True)
                print("-> Dumped statistics to {}".format(stats_fname))
    else:
        dataset.set_partition("default")
        dataset.set_split(None)
        print("Computing single statistics for whole dataset")
        _band_values, band_stats = compute_dataset_statistics(
            dataset, n_value_per_image=values_per_image, n_samples=samples
        )
        stats_fname = os.path.join(dataset.dataset_dir, "band_stats.json")
        with open(stats_fname, "w", encoding="utf8") as fp:
            json.dump({k: v.to_dict() for k, v in band_stats.items()}, fp)
        print("Statistics written to {stats_fname}.")


def start():
    args = parser.parse_args()
    produce_band_stats(args.dataset, args.splits, args.values_per_image, args.samples)


if __name__ == "__main__":
    start()
    # produce_band_stats(io.CCB_DIR / "converted" / "eurosat")
