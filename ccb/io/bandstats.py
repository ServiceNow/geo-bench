'''
Compute dataset band statistics for each band and save them in bandstats.json

For the future, implement partitions and splits
'''
import argparse
import json
import os

from ccb.io.dataset import Dataset, dataset_statistics


parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='path to CCB dataset folder')


def main(args):
    print('Loading dataset', args.dataset)
    dataset = Dataset(args.dataset)
    print(dataset)
    print('Computing statistics')
    stats = dataset_statistics
    stats_fname = os.path.join(args.dataset, 'bandstats.json')
    with open(stats_fname, 'wb') as fp:
        json.dump(stats, fp)




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)