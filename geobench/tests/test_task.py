import json
import pickle

import numpy as np
import pytest
import rasterio

import geobench as gb
from geobench.task import GEO_BENCH_DIR, load_task_specs


def write_dataset(dataset_dir, sample_name="sample1"):
    """Write a one-sample dataset, with its task specs and default partition."""
    dataset_dir.mkdir(parents=True)

    bands = [
        gb.Band(
            np.random.randint(1, 1000, (4, 4), dtype=np.int16).astype(float),
            gb.SpectralBand(name, alt_names=(f"alt_{name}",), spatial_resolution=20),
            10,
            transform=rasterio.transform.from_bounds(1, 2, 3, 3, 4, 5),
            crs="EPSG:4326",
        )
        for name in ("red", "green", "blue")
    ]
    sample = gb.Sample(bands, np.random.randint(2), sample_name)
    sample.write(str(dataset_dir))

    task_specs = gb.TaskSpecifications(
        dataset_name="wrong_name",
        benchmark_name="wrong_benchmark",
        patch_size=(4, 4),
        spatial_resolution=1.0,
        bands_info=[band.band_info for band in bands],
    )
    task_specs.save(str(dataset_dir), overwrite=True)

    partition = gb.Partition()
    partition.add("train", sample.sample_name)
    partition.save(directory=str(dataset_dir), partition_name="default")


@pytest.fixture
def benchmark_dir(tmp_path):
    """A benchmark with two datasets, outside of $GEO_BENCH_DIR."""
    benchmark_dir = tmp_path / "my_benchmark"
    for dataset_name in ("m-one", "m-two"):
        write_dataset(benchmark_dir / dataset_name)
    with open(benchmark_dir / "m-one" / "label_map.json", "w") as fd:
        json.dump({"label": ["path"]}, fd)
    return benchmark_dir


def test_task_iterator_reads_datasets_from_benchmark_dir(benchmark_dir):
    tasks = {task.dataset_name: task for task in gb.task_iterator(benchmark_dir=benchmark_dir)}
    assert sorted(tasks) == ["m-one", "m-two"]

    task = tasks["m-one"]
    assert task.benchmark_name == "my_benchmark"
    assert task.get_dataset_dir() == benchmark_dir / "m-one"
    assert task.get_label_map() == {"label": ["path"]}

    dataset = task.get_dataset(split="train")
    assert len(dataset) == 1
    assert len(dataset[0].bands) == 3


def test_get_dataset_dir_defaults_to_geo_bench_dir():
    task_specs = gb.TaskSpecifications(
        dataset_name="m-one",
        benchmark_name="my_benchmark",
        patch_size=(4, 4),
        spatial_resolution=1.0,
        bands_info=[],
    )
    assert task_specs.get_dataset_dir() == GEO_BENCH_DIR / "my_benchmark" / "m-one"


def test_saved_task_specs_do_not_carry_the_dataset_dir(benchmark_dir, tmp_path):
    """A shipped task_specs.pkl must not pin the directory it was written from."""
    task_specs = load_task_specs(benchmark_dir / "m-one")
    assert task_specs.dataset_dir == benchmark_dir / "m-one"

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    task_specs.save(str(elsewhere))
    with open(elsewhere / "task_specs.pkl", "rb") as fd:
        assert pickle.load(fd).dataset_dir is None


def test_ignore_task_excludes_the_named_tasks(benchmark_dir):
    tasks = gb.task_iterator(benchmark_dir=benchmark_dir, ignore_task=["m-two"])
    assert [task.dataset_name for task in tasks] == ["m-one"]
