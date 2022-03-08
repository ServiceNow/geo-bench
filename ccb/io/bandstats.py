"""
Compute dataset band statistics for each band and save them in bandstats.json

For the future, implement partitions and splits
"""
import argparse
import json
import os

from ccb.io.dataset import Dataset, dataset_statistics, dataset_statistics2


parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="path to CCB dataset folder")


def main(args):
    print("Loading dataset", args.dataset)
    dataset = Dataset(args.dataset)
    print(dataset)
    for partition in dataset.list_partitions():
        dataset.set_active_partition(partition)
        for split in dataset.list_splits():
            print(f"Computing statistics for {partition}:{split}")
            stats = dataset_statistics2(dataset, n_value_per_image=1000, n_samples=100)
            stats_fname = os.path.join(args.dataset, f"{partition}_{split}_bandstats.json")
            with open(stats_fname, "w", encoding="utf8") as fp:
                json.dump(stats, fp)
            print("-> Dumped statistics to {}".format(stats_fname))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
