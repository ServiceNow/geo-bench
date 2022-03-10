"""
Compute dataset band statistics for each band and save them in bandstats.json

For the future, implement partitions and splits
"""
import argparse
import json
import os

from ccb.io.dataset import Dataset, dataset_statistics


parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="path to CCB dataset folder")
parser.add_argument("-s", "--splits", default=False, help="compute statistics separately for each split")
parser.add_argument("-c", "--check", default=False, help="don't compute statistics, but load and check only")


def main(args):
    print("Loading dataset", args.dataset)
    dataset = Dataset(args.dataset)
    print(dataset)
    if not args.check:
        if args.splits:
            for partition in dataset.list_partitions():
                dataset.set_active_partition(partition)
                for split in dataset.list_splits():
                    print(f"Computing statistics for {partition}:{split}")
                    dataset.set_split(split)
                    band_values, band_stats = dataset_statistics(dataset, n_value_per_image=1000, n_samples=100)
                    stats_fname = os.path.join(args.dataset, f"{partition}_{split}_bandstats.json")
                    with open(stats_fname, "w", encoding="utf8") as fp:
                        json.dump({k:v.to_dict() for k,v in band_stats.items()}, fp)
                    print("-> Dumped statistics to {}".format(stats_fname))
        else:
            dataset.set_active_partition('default')
            dataset.set_split(None)
            print(f"Computing single statistics for whole dataset")
            band_values, band_stats = dataset_statistics(dataset, n_value_per_image=1000, n_samples=100)
            stats_fname = os.path.join(args.dataset, f"all_bandstats.json")
            with open(stats_fname, "w", encoding="utf8") as fp:
                json.dump({k:v.to_dict() for k,v in band_stats.items()}, fp)
            print("-> Dumped statistics to {}".format(stats_fname))
        






if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
