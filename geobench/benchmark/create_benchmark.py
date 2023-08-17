"""Create benchmark."""
import shutil
from collections import defaultdict
from locale import nl_langinfo
from math import floor
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm

from geobench import io
from geobench.io import bandstats
from geobench.benchmark.generate_partitions import generate_train_size_sweep

def make_subsampler(max_sizes):
    """Create a subsampler.

    Args:
        max_sizes:

    Returns:
        callable subsampler
    """

    def _subsample(partition, task_specs, rng=np.random):
        return subsample(partition=partition, max_sizes=max_sizes, rng=rng)

    return _subsample


def subsample(partition: io.Partition, max_sizes: Dict[str, int], rng=np.random) -> io.Partition:
    """Randomly subsample `partition` to satisfy `max_sizes`.

    Args:
        partition:
        max_sizes:
        rng:

    Returns:
        subsampled partition
    """
    new_partition = io.Partition()

    for split_name, sample_names in partition.partition_dict.items():
        if len(sample_names) > max_sizes[split_name]:
            subset = list(rng.choice(sample_names, max_sizes[split_name], replace=False))
        else:
            subset = sample_names[:]  # create a copy to avoid potential issues
        new_partition.partition_dict[split_name] = subset
    return new_partition


def _make_split_label_maps(
    label_map: Dict[int, List[str]], partition_dict: Dict[str, List[str]]
) -> Dict[str, Dict[int, List[str]]]:
    """Organize label map into 'train', 'valid' and 'test'.

    Args:
        label_map:
        partition_dict:

    Retruns:
        split label map
    """
    split_label_maps: Dict[str, Dict[int, List[str]]] = {}
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


def _filter_for_min_size(split_label_maps, min_class_sizes: Dict[str, int]) -> DefaultDict[str, Dict[int, List[str]]]:
    """Make sure each class has statisfies `min_class_sizes`.

    Args:
        split_label_maps:
        min_class_sizes:
    """
    new_split_label_maps: DefaultDict[str, Dict[int, List[str]]] = defaultdict(dict)
    for label in split_label_maps["train"].keys():

        ok = True
        for split, min_class_size in min_class_sizes.items():
            if len(split_label_maps[split].get(label, ())) < min_class_size:
                ok = False
        if ok:
            for split in ("train", "valid", "test"):
                new_split_label_maps[split][label] = split_label_maps[split][label][:]

    return new_split_label_maps


def assert_no_overlap(split_label_maps: Dict[str, Dict[int, List[str]]]) -> None:
    """Asser that label map is a partition and that no sample are common across splits.

    Args:
        split_label_maps:
    """
    sample_set = set()
    total_count = 0
    for label_map in split_label_maps.values():
        for sample_names in label_map.values():
            sample_set.update(sample_names)
            total_count += len(sample_names)

    assert len(sample_set) == total_count


def make_resampler(max_sizes, min_class_sizes: Dict[str, int] = {"train": 10, "valid": 1, "test": 1}):
    """Matrialize a resampler with the required interface.

    Args:
        max_sizes:
        min_class_sizes:

    Returns:
        callable resampling function
    """

    def _resample(partition, task_specs, rng=np.random):
        label_map = task_specs.get_label_map()
        return resample(
            partition=partition, label_map=label_map, max_sizes=max_sizes, min_class_sizes=min_class_sizes, rng=rng
        )

    return _resample


def resample(
    partition: io.Partition,
    label_map: Dict[int, List[str]],
    max_sizes: Dict[str, int],
    min_class_sizes: Dict[str, int],
    verbose: bool = True,
    rng=np.random,
) -> io.Partition:
    """Reduce class imbalance in `partition` based on information in `label_map`.

    Args:
        partition:
        label_map:
        max_sizes:
        min_class_sizes:
        verbose:
        rng:

    Returns:
        resampled partition
    """
    split_label_maps = _make_split_label_maps(label_map, partition_dict=partition.partition_dict)
    assert_no_overlap(split_label_maps)
    new_split_label_maps = _filter_for_min_size(split_label_maps, min_class_sizes)
    assert_no_overlap(new_split_label_maps)
    partition_dict = defaultdict(list)
    for split in ("train", "valid", "test"):

        label_map = new_split_label_maps[split]
        n_classes = len(label_map)
        if split == "train":
            # aim for uniform distribution
            class_distribution: Dict[int, float] = {label: 1 / n_classes for label in label_map.keys()}

        for label, sample_names in label_map.items():
            max_sample = floor(max_sizes[split] * class_distribution[label])

            if len(sample_names) > max_sample:
                label_map[label] = rng.choice(sample_names, size=max_sample, replace=False)

            partition_dict[split].extend(label_map[label])

        if split == "train":
            total_sizes = sum([len(sample_names) for sample_names in label_map.values()])
            class_distribution = {label: len(sample_names) / total_sizes for label, sample_names in label_map.items()}

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


def make_resampler_from_stats(max_sizes):
    """Matrialize a resampler with the required interface."""

    def _resample(partition, task_specs, rng=np.random):
        label_stats = task_specs.label_stats()
        return resample_from_stats(partition=partition, label_stats=label_stats, max_sizes=max_sizes, rng=rng)

    return _resample


def resample_from_stats(
    partition: io.Partition,
    label_stats: Dict[str, List[float]],
    max_sizes: Dict[str, int],
    verbose: bool = True,
    rng=np.random,
    return_prob: bool = False,
) -> Union[io.Partition, Tuple[io.Partition, Dict[str, List[str]]]]:
    """Resample based on statistics.

    Args:
        partition:
        label_stats:
        max_sizes:
        verbose:
        rng:
        return_prob:

    Returns:
        resampled partition
    """
    partition_dict = defaultdict(list)
    prob_dict: Dict[str, List[str]] = {}
    for split, sample_names in partition.partition_dict.items():

        if len(sample_names) > max_sizes[split]:

            stats = np.array([label_stats[sample_name] for sample_name in sample_names])
            cum_stats = np.sum(stats, axis=0, keepdims=True)
            weight_factors = 1 / (cum_stats + 1)
            prob = np.sum(stats * weight_factors, axis=1)
            prob /= prob.sum()

            partition_dict[split] = list(rng.choice(sample_names, size=max_sizes[split], replace=False, p=prob))
            prob_dict[split] = prob

        else:
            print(f"Split {split} unchanged since {len(sample_names)} <= {max_sizes[split]}.")
            partition_dict[split] = sample_names

    new_partition = io.Partition(partition_dict=partition_dict)

    if return_prob:
        return new_partition, prob_dict
    else:
        return new_partition


def max_shape_center_crop(max_shape):
    """Ensure that the largest band has `max_shape` or less.

    If not, all bands will be center-cropped proportionnally
    e.g., a band that is half the size of the max band will have a crop that is half the size of max_shape.
    """
    max_shape = np.array(max_shape)

    def sample_converter(sample: io.Sample) -> io.Sample:

        # # include label if it is a band
        # band_label = isinstance(sample.label, io.Band)
        # bands: List[Any] = sample.bands
        # if band_label:
        #     bands.append(sample.label)

        # find max shape
        max_band_shape = np.array(sample.largest_shape())

        # nothing to do in that case
        if np.all(max_band_shape <= np.array(max_shape)):
            return sample

        elif np.all(max_band_shape > np.array(max_shape)):

            size_ratio = max_shape / max_band_shape
            start_ratio = (1.0 - size_ratio) / 2.0

            for band in sample.bands:
                band.crop_from_ratio(start_ratio, size_ratio)

            if isinstance(sample.label, io.Band):
                sample.label.crop_from_ratio(start_ratio, size_ratio)

            return sample

        else:
            raise ValueError(
                "`max_shape` has one dimension smaller and one dimension bigger than then the max shape of all bands."
            )

    return sample_converter


def transform_dataset(
    dataset_dir: Path,
    new_benchmark_dir: Path,
    partition_name: str,
    resampler=None,
    sample_converter=None,
    delete_existing: bool = False,
    hdf5: bool = True,
) -> Union[Path, None]:
    """Transform dataset.

    Args:
        dataset_dir:
        new_benchmark_dir:
        partition_name:
        resample:
        sample_converter:
        delete_existing:
        hdf5:
    """
    dataset = io.GeobenchDataset(dataset_dir, partition_name=partition_name)
    task_specs = dataset.task_specs
    task_specs.benchmark_name = dataset_dir.parent.name
    new_dataset_dir = new_benchmark_dir / dataset_dir.name

    if new_dataset_dir.exists():
        if delete_existing:
            print(f"Deleting exising dataset {new_dataset_dir}.")
            shutil.rmtree(new_dataset_dir)
        else:
            print(f"Skipping {new_dataset_dir} it already exists.")
            return None

    new_dataset_dir.mkdir(parents=True, exist_ok=True)

    if resampler is not None:
        new_partition = resampler(partition=dataset.load_partition(partition_name), task_specs=task_specs)
    else:
        new_partition = dataset.load_partition(partition_name)

    task_specs.benchmark_name = new_benchmark_dir.name
    task_specs.save(new_dataset_dir, overwrite=True)

    # TODO task_specs should be updated if sample_converter modifies the patch_size.

    for split_name, sample_names in new_partition.partition_dict.items():
        print(f"  Converting {len(sample_names)} samples from {split_name} split.")
        for sample_name in tqdm(sample_names):
            if hdf5:
                sample_name += ".hdf5"

            if sample_converter is None:
                if hdf5:
                    shutil.copyfile(dataset_dir / sample_name, new_dataset_dir / sample_name)
                else:
                    shutil.copytree(dataset_dir / sample_name, new_dataset_dir / sample_name, dirs_exist_ok=True)
            else:
                format = "hdf5" if hdf5 else "tif"
                sample = io.load_sample(dataset_dir / sample_name, format=format)
                new_sample = sample_converter(sample)
                new_sample.write(new_dataset_dir, format=format)

    new_partition.save(new_dataset_dir, "default")
    return new_dataset_dir


def _make_benchmark(new_benchmark_name, specs, src_benchmark_name="converted"):
    """Create benchmark."""
    for dataset_name, (resampler, sample_converter) in specs.items():
        print(f"Transforming {dataset_name}.")
        dataset_dir = io.GEO_BENCH_DIR / src_benchmark_name / dataset_name
        new_dataset_dir = transform_dataset(
            dataset_dir=dataset_dir,
            new_benchmark_dir=io.GEO_BENCH_DIR / new_benchmark_name,
            partition_name="default",
            resampler=resampler,
            sample_converter=sample_converter,
            delete_existing=False,
        )

        if new_dataset_dir is not None:
            print(f"  Producing band stats for {dataset_name}.")
            dataset = io.GeobenchDataset(new_dataset_dir)
            bandstats.produce_band_stats(dataset)

            print(f"  Producing partitions for {dataset_name}.")
            print(f"    Using partition {dataset.active_partition_name} in directory {dataset.dataset_dir}.")

            generate_train_size_sweep(
                partition=dataset.active_partition,
                train_fractions=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
                dataset_dir=dataset.dataset_dir,
            )

            print()


def make_classification_benchmark():
    """Enrtypoint for creating the classification benchmark."""
    max_sizes = {"train": 20000, "valid": 1000, "test": 1000}
    # max_sizes = {"train": 10, "valid": 100, "test": 100}
    default_resampler = make_resampler(max_sizes=max_sizes)
    specs = {
        # "forestnet_v1.0": (default_resampler, max_shape_center_crop((256, 256))),
        "eurosat": (default_resampler, None),
        # "brick_kiln_v1.0": (default_resampler, None),
        # "so2sat": (default_resampler, None),
        # "pv4ger_classification": (default_resampler, max_shape_center_crop((256, 256))),
        # # "geolifeclef-2021": (make_resampler(max_sizes={"train": 10000, "valid": 5000, "test": 5000}), None),
        # "geolifeclef-2022": (default_resampler, None),
        # "bigearthnet": (make_resampler_from_stats(max_sizes), None),
    }
    _make_benchmark("classification_v0.8.2", specs)


def make_segmentation_benchmark():
    """Create segmentation benchmark."""
    max_sizes = {"train": 3000, "valid": 1000, "test": 1000}
    # default_resampler = make_subsampler(max_sizes=max_sizes)
    resampler_from_stats = make_resampler_from_stats(max_sizes=max_sizes)
    specs = {
        # "pv4ger_segmentation": (resampler_from_stats, None),
        # "xview2": (resampler_from_stats, None),
        # # "forestnet_v1.0": (resampler_from_stats, None),
        # "cvpr_chesapeake_landcover": (resampler_from_stats, None),
        # "smallholder_cashew": (resampler_from_stats, None)
        # "southAfricaCropType": (resampler_from_stats, None)
        # "nz_cattle_segmentation": (resampler_from_stats, None),
        "NeonTree_segmentation": (resampler_from_stats, None),
        # "seasonet": (resampler_from_stats, None),
    }
    _make_benchmark("segmentation_v0.3", specs)


if __name__ == "__main__":
    make_classification_benchmark()
    # make_segmentation_benchmark()


# procedure
# * make sure label_map.py was executed for the considered dataset (ensure benchmark_name set to "converted")
# * create your benchmark here
# * run label_map.py again to create band_stats (make sure that benchmark_name is set to the newly created benchmark, also make sure that compute_band_stats=True )
