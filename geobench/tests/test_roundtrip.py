"""End-to-end round trip of every geobench storage format."""

import datetime
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio

import geobench as gb
from geobench import dataset as dataset_module


def make_sample(name="s0", seg=False):
    transform = rasterio.transform.from_bounds(1, 2, 3, 3, 16, 16)
    bands = []
    for band_info in gb.sentinel2_13_bands[:3]:
        data = np.random.randint(1, 1000, (16, 16), dtype=np.int16)
        bands.append(
            gb.Band(
                data,
                band_info,
                10,
                transform=transform,
                crs="EPSG:4326",
                date=datetime.date(2020, 1, 1),
                meta_info={"cloud": 0.1, "arr": np.arange(3)},
            )
        )

    label: object
    if seg:
        label = gb.Band(
            np.random.randint(0, 4, (16, 16)).astype(np.int16),
            gb.SegmentationClasses(
                "label", spatial_resolution=10, n_classes=4, class_names=["a", "b", "c", "d"]
            ),
            10,
            transform=transform,
            crs="EPSG:4326",
        )
    else:
        label = 2

    return gb.Sample(bands, label, name)


def check(sample, sample_):
    assert sample.sample_name == sample_.sample_name
    assert len(sample.bands) == len(sample_.bands)
    for band, band_ in zip(
        sorted(sample.bands, key=lambda b: b.band_info.name),
        sorted(sample_.bands, key=lambda b: b.band_info.name),
    ):
        assert band.band_info == band_.band_info
        np.testing.assert_array_equal(band.data, band_.data)
        assert band.date == band_.date

    image, dates, band_names = sample_.pack_to_4d(resample=True)
    sample_.pack_to_3d(band_names=tuple(band_names), resample=True)
    assert image.shape[0] == len(dates)


@pytest.mark.parametrize("seg", [False, True])
@pytest.mark.parametrize(
    "writer,loader",
    [
        (dataset_module.write_sample_hdf5, dataset_module.load_sample_hdf5),
        (dataset_module.write_sample_npz, dataset_module.load_sample_npz),
    ],
)
def test_roundtrip_hdf5_npz(writer, loader, seg):
    with tempfile.TemporaryDirectory() as dataset_dir:
        sample = make_sample(seg=seg)
        sample_ = loader(Path(writer(sample, dataset_dir)))
        check(sample, sample_)

        if seg:
            assert isinstance(sample_.label, gb.Band)
        else:
            assert sample_.label == 2


@pytest.mark.parametrize("seg", [False, True])
def test_roundtrip_tif(seg):
    with tempfile.TemporaryDirectory() as dataset_dir:
        sample = make_sample(seg=seg)
        band_names = [band.band_info.name for band in sample.bands]
        sample.write(dataset_dir, format="tif")
        sample_ = dataset_module.load_sample_tif(
            Path(dataset_dir) / sample.sample_name, band_names=band_names
        )
        check(sample, sample_)


def test_dataset_roundtrip_and_statistics():
    with tempfile.TemporaryDirectory() as dataset_dir:
        samples = [make_sample(f"s{i}", seg=True) for i in range(4)]
        for sample in samples:
            sample.write(dataset_dir, format="hdf5")

        bands_info = gb.sentinel2_13_bands[:3]
        task_specs = gb.TaskSpecifications(
            dataset_name="test",
            benchmark_name="bench",
            patch_size=(16, 16),
            spatial_resolution=10.0,
            bands_info=bands_info,
            label_type=gb.SegmentationClasses(
                "label", spatial_resolution=10, n_classes=4, class_names=["a", "b", "c", "d"]
            ),
            n_time_steps=1,
        )
        task_specs.save(dataset_dir, overwrite=True)

        partition = gb.Partition()
        for split, sample in zip(["train", "train", "valid", "test"], samples):
            partition.add(split, sample.sample_name)
        partition.save(directory=dataset_dir, partition_name="default")

        dataset = gb.GeobenchDataset(
            dataset_dir,
            band_names=[band_info.name for band_info in bands_info],
            partition_name="default",
            format="hdf5",
        )
        dataset.set_split("train")
        assert len(dataset) == 2

        task_specs_ = gb.task.load_task_specs(Path(dataset_dir), rename_benchmark=False)
        assert task_specs_.dataset_name == "test"

        _, band_stats = dataset_module.compute_dataset_statistics(dataset, n_value_per_image=10)
        assert set(band_stats) >= {band_info.name for band_info in bands_info}
