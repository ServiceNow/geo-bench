"""
Compute dataset band statistics for each band and save them in bandstats.json

For the future, implement partitions and splits
"""
import argparse
import json
import os

from ccb.io.dataset import Dataset, compute_dataset_statistics


parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="path to CCB dataset folder")
parser.add_argument("-s", "--splits", default=False, help="compute statistics separately for each split")
parser.add_argument("-c", "--check-only", default=True, help="don't compute statistics, but load and check only")
parser.add_argument("--values-per-image", default=1000, help="values per image")
parser.add_argument("--samples", default=1000, help="dataset subset size")


def bandstats(dataset_dir, use_splits, values_per_image, samples):

    dataset = Dataset(dataset_dir)

    if use_splits:
        for partition in dataset.list_partitions():
            dataset.set_partition(partition)
            for split in dataset.list_splits():
                print(f"Computing statistics for {partition}:{split}")
                dataset.set_split(split)
                band_values, band_stats = compute_dataset_statistics(
                    dataset, n_value_per_image=values_per_image, n_samples=samples
                )
                stats_fname = os.path.join(dataset.dataset_dir, f"{partition}_{split}_bandstats.json")
                with open(stats_fname, "w", encoding="utf8") as fp:
                    json.dump({k: v.to_dict() for k, v in band_stats.items()}, fp)
                print("-> Dumped statistics to {}".format(stats_fname))
    else:
        dataset.set_partition("default")
        dataset.set_split(None)
        print("Computing single statistics for whole dataset")
        band_values, band_stats = compute_dataset_statistics(
            dataset, n_value_per_image=values_per_image, n_samples=samples
        )
        stats_fname = os.path.join(dataset.dataset_dir, "all_bandstats.json")
        with open(stats_fname, "w", encoding="utf8") as fp:
            json.dump({k: v.to_dict() for k, v in band_stats.items()}, fp)
        print("-> Dumped statistics to {}".format(stats_fname))


def main(args):
    print("Loading dataset", args.dataset)
    dataset = Dataset(args.dataset)
    print(dataset)
    if not args.check_only:
        bandstats(dataset, args.splits, args.values_per_image, args.samples)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
