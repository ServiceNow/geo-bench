import tempfile

import numpy as np
import pytest
import rasterio

import geobench as gb


def random_band(shape=(16, 16), band_name="test_band", alt_band_names=("alt_name",)):
    data = np.random.randint(1, 1000, shape, dtype=np.int16).astype(float)
    data *= 2.1
    if len(shape) == 3 and shape[2] > 1:
        band_info = gb.MultiBand(
            band_name, alt_names=alt_band_names, spatial_resolution=20, n_bands=shape[2]
        )
    else:
        band_info = gb.SpectralBand(
            band_name, alt_names=alt_band_names, spatial_resolution=20, wavelength=0.1
        )

    transform = rasterio.transform.from_bounds(1, 2, 3, 3, 4, 5)

    return gb.Band(data, band_info, 10, transform=transform, crs="EPSG:4326")


def random_sample(n_bands=3, name="test_sample"):
    bands = [
        random_band(band_name=f"{i:2d}", alt_band_names=(f"alt_{i:2d}")) for i in range(n_bands)
    ]
    return gb.Sample(bands, np.random.randint(2), name)


def test_pack_4d_dense():
    bands = [
        random_band((3, 4), "band_1", ("alt_band_1",)),
        random_band((3, 4), "band_2", ("alt_band_2",)),
        random_band((6, 8), "band_3", ("alt_band_3",)),
    ]
    sample = gb.Sample(bands, np.random.randint(2), "test_sample")
    image, dates, band_names = sample.pack_to_4d(
        resample=True, band_names=("band_1", "band_2", "band_3")
    )
    image_, _ = sample.pack_to_3d(resample=True, band_names=("band_1", "band_2", "band_3"))

    np.testing.assert_array_equal(image[0], image_)

    assert image.shape == (1, 6, 8, 3)
    assert dates == [None]
    assert tuple(band_names) == ("band_1", "band_2", "band_3")

    image, dates, band_names = sample.pack_to_4d(band_names=("band_1", "band_2"))

    assert image.shape == (1, 3, 4, 2)
    assert dates == [None]
    assert tuple(band_names) == ("band_1", "band_2")


def test_crop_from_ratio():
    band = random_band(shape=(10, 10))
    old_data = band.data
    band.crop_from_ratio((0.1, 0.1), (0.8, 0.8))
    assert band.data.shape == (8, 8)
    np.testing.assert_equal(old_data[1:9, 1:9], band.data)


def test_pack_4d_multi_band():
    bands = [
        random_band((3, 4, 5), "band_1", ("alt_band_1",)),
        random_band((3, 4), "band_2", ("alt_band_2",)),
        random_band((6, 8), "band_3", ("alt_band_3",)),
    ]
    sample = gb.Sample(bands, np.random.randint(2), "test_sample")
    image, dates, band_names = sample.pack_to_4d(resample=True)

    assert dates == [None]
    assert image.shape == (1, 6, 8, 7)
    assert tuple(band_names) == ("band_1",) * 5 + ("band_2", "band_3")


def test_write_read():
    with tempfile.TemporaryDirectory() as dataset_dir:
        sample = random_sample()
        sample.write(dataset_dir)
        band_names = [band.band_info.name for band in sample.bands]
        # define task_spec for dataset

        bands_info = [
            gb.SpectralBand(
                name=band.band_info.name,
                alt_names=(band.band_info.alt_names,),
                spatial_resolution=band.band_info.spatial_resolution,
            )
            for band in sample.bands
        ]

        task_specs = gb.TaskSpecifications(
            dataset_name="test",
            benchmark_name="test_bench",
            patch_size=(16, 16),
            spatial_resolution=1.0,
            bands_info=bands_info,
        )
        task_specs.save(dataset_dir, overwrite=True)

        partition = gb.Partition()
        partition.add("train", sample.sample_name)
        partition.save(directory=dataset_dir, partition_name="default")
        ds = gb.GeobenchDataset(dataset_dir, band_names=band_names, partition_name="default")
        sample_ = list(ds.iter_dataset(1))[0]

    assert len(sample.bands) == len(sample_.bands)
    # TODO need to review test here
    for band in sample.bands:
        assert len(list(filter(lambda band_: band.band_info == band_.band_info, sample_.bands))) > 0
        # assert len(list(filter(lambda band_: band.crs == band_.crs, sample_.bands))) > 0


def assert_same_sample(sample, sample_):
    assert sample.sample_name == sample_.sample_name
    assert len(sample.bands) == len(sample_.bands)
    for band in sample.bands:
        len(list(filter(lambda band_: band.band_info == band_.band_info, sample_.bands))) > 0


def test_dataset_partition():
    # Create fake dataset
    with tempfile.TemporaryDirectory() as dataset_dir:
        sample1 = random_sample(name="sample1")
        sample1.write(dataset_dir)

        sample2 = random_sample(name="sample2")
        sample2.write(dataset_dir)

        sample3 = random_sample(name="sample3")
        sample3.write(dataset_dir)

        band_names = [band.band_info.name for band in sample1.bands]

        bands_info = [
            gb.SpectralBand(
                name=band.band_info.name,
                alt_names=(band.band_info.alt_names,),
                spatial_resolution=band.band_info.spatial_resolution,
            )
            for band in sample1.bands
        ]

        task_specs = gb.TaskSpecifications(
            dataset_name="test",
            benchmark_name="test_bench",
            patch_size=(16, 16),
            spatial_resolution=1.0,
            bands_info=bands_info,
        )
        task_specs.save(dataset_dir, overwrite=True)

        # Create default partition
        partition = gb.Partition()
        partition.add("train", sample1.sample_name)
        partition.add("valid", sample2.sample_name)
        partition.add("valid", sample3.sample_name)
        partition.save(directory=dataset_dir, partition_name="default")

        # Create funky partition
        partition = gb.Partition()
        partition.add("valid", sample1.sample_name)
        partition.add("test", sample2.sample_name)
        partition.add("train", sample3.sample_name)
        partition.save(directory=dataset_dir, partition_name="funky")

        # Test 1: load partition default, no split
        ds = gb.GeobenchDataset(dataset_dir, band_names=band_names, partition_name="default")
        assert set(ds.list_partitions()) == set(["funky", "default"])
        assert ds.active_partition_name == "default"  # use default normally
        assert set(ds.list_splits()) == set(["train", "valid", "test"])
        assert ds.split is None
        assert len(ds) == 3

        # Ordering is not guaranteed. Do we want to enforce that? The following can fail
        # assert_same_sample(ds[0], sample1)
        # assert_same_sample(ds[1], sample2)
        # assert_same_sample(ds[2], sample3)

        ds.set_split("train")
        assert ds.split == "train"
        assert_same_sample(ds[0], sample1)
        assert len(ds) == 1

        ds.set_split("valid")
        assert ds.split == "valid"
        # Try strict ordering
        try:
            assert_same_sample(ds[0], sample2)
            assert_same_sample(ds[1], sample3)
        except Exception:
            assert_same_sample(ds[0], sample3)
            assert_same_sample(ds[1], sample2)
        assert len(ds) == 2

        ds.set_split("test")
        assert ds.split == "test"
        assert len(ds) == 0
        with pytest.raises(IndexError):  # default:test is empty
            ds[0]

        ds = gb.GeobenchDataset(dataset_dir, band_names=band_names, partition_name="funky")
        assert set(ds.list_partitions()) == set(["funky", "default"])
        assert ds.active_partition_name == "funky"  # use default normally
        assert set(ds.list_splits()) == set(["train", "valid", "test"])
        assert len(ds) == 3

        ds.set_split("train")
        assert ds.split == "train"
        assert_same_sample(ds[0], sample3)
        assert len(ds) == 1

        ds.set_split("valid")
        assert ds.split == "valid"
        assert_same_sample(ds[0], sample1)
        assert len(ds) == 1

        ds.set_split("test")
        assert ds.split == "test"
        assert_same_sample(ds[0], sample2)
        assert len(ds) == 1
        with pytest.raises(IndexError):  # default:test is out of bounds
            ds[2]


def test_dataset_withnopartition():
    with tempfile.TemporaryDirectory() as dataset_dir:
        sample1 = random_sample(name="sample1")
        sample1.write(dataset_dir)

        sample2 = random_sample(name="sample2")
        sample2.write(dataset_dir)

        sample3 = random_sample(name="sample3")
        sample3.write(dataset_dir)

        band_names = [band.band_info.name for band in sample1.bands]

        with pytest.raises(ValueError):  # raise ValueError because not partition exists
            _ = gb.GeobenchDataset(dataset_dir, band_names=band_names, partition_name="default")


def test_class_id():
    from geobench import dataset

    assert isinstance(dataset.sentinel2_13_bands[0], gb.SpectralBand)
    assert isinstance(gb.sentinel2_13_bands[0], dataset.SpectralBand)
