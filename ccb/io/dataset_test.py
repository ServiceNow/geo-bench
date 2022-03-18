from ccb import io
import numpy as np
import tempfile
from ccb.io.bandstats import bandstats
import pytest


def random_band(shape=(16, 16), band_name="test_band"):
    data = np.random.randint(1, 1000, shape, dtype=np.int16).astype(np.float)
    data *= 2.1
    if len(shape) == 3 and shape[2] > 1:
        band_info = io.MultiBand(band_name, alt_names=("tb"), spatial_resolution=20, n_bands=shape[2])
    else:
        band_info = io.SpectralBand(band_name, alt_names=("tb"), spatial_resolution=20, wavelength=0.1)
    return io.Band(data, band_info, 10)


def random_sample(n_bands=3, name="test_sample"):
    bands = [random_band(band_name=f"{i:2d}") for i in range(n_bands)]
    return io.Sample(bands, np.random.randint(2), name)


def test_pack_4d_dense():
    bands = [random_band((3, 4), "band_1"), random_band((3, 4), "band_2"), random_band((6, 8), "band_3")]
    sample = io.Sample(bands, np.random.randint(2), "test_sample")
    image, dates, band_names = sample.pack_to_4d(resample=True)
    image_, _ = sample.pack_to_3d(resample=True)

    np.testing.assert_array_equal(image[0], image_)

    assert image.shape == (1, 6, 8, 3)
    assert dates == [None]
    assert tuple(band_names) == ("band_1", "band_2", "band_3")

    image, dates, band_names = sample.pack_to_4d(band_names=("band_1", "band_2"))

    assert image.shape == (1, 3, 4, 2)
    assert dates == [None]
    assert tuple(band_names) == ("band_1", "band_2")


def test_pack_4d_multi_band():
    bands = [random_band((3, 4, 5), "band_1"), random_band((3, 4), "band_2"), random_band((6, 8), "band_3")]
    sample = io.Sample(bands, np.random.randint(2), "test_sample")
    image, dates, band_names = sample.pack_to_4d(resample=True)

    assert dates == [None]
    assert image.shape == (1, 6, 8, 7)
    assert tuple(band_names) == ("band_1",) * 5 + ("band_2", "band_3")


def test_write_read():
    with tempfile.TemporaryDirectory() as dataset_dir:
        sample = random_sample()
        sample.write(dataset_dir)
        partition = io.Partition()
        partition.add("train", sample.sample_name)
        partition.save(directory=dataset_dir, partition_name="default")
        ds = io.Dataset(dataset_dir)
        sample_ = list(ds.iter_dataset(1))[0]

    assert len(sample.bands) == len(sample_.bands)
    for band in sample.bands:
        len(list(filter(lambda band_: band.band_info == band_.band_info, sample_.bands))) > 0


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

        # Create default partition
        partition = io.Partition()
        partition.add("train", sample1.sample_name)
        partition.add("valid", sample2.sample_name)
        partition.add("valid", sample3.sample_name)
        partition.save(directory=dataset_dir, partition_name="default")

        # Create funky partition
        partition = io.Partition()
        partition.add("valid", sample1.sample_name)
        partition.add("test", sample2.sample_name)
        partition.add("train", sample3.sample_name)
        partition.save(directory=dataset_dir, partition_name="funky")

        # Test 1: load partition default, no split
        ds = io.Dataset(dataset_dir)
        assert set(ds.list_partitions()) == set(["funky", "default"])
        assert ds.get_partition() == "default"  # use default normally
        assert set(ds.list_splits()) == set(["train", "valid", "test"])
        assert ds.get_split() is None
        assert len(ds) == 3
        # Ordering is not guaranteed. Do we want to enforce that? The following can fail
        # assert_same_sample(ds[0], sample1)
        # assert_same_sample(ds[1], sample2)
        # assert_same_sample(ds[2], sample3)

        ds.set_split("train")
        assert ds.get_split() == "train"
        assert_same_sample(ds[0], sample1)
        assert len(ds) == 1

        ds.set_split("valid")
        assert ds.get_split() == "valid"
        # Try strict ordering
        try:
            assert_same_sample(ds[0], sample2)
            assert_same_sample(ds[1], sample3)
        except Exception:
            assert_same_sample(ds[0], sample3)
            assert_same_sample(ds[1], sample2)
        assert len(ds) == 2

        ds.set_split("test")
        assert ds.get_split() == "test"
        assert len(ds) == 0
        with pytest.raises(IndexError):  # default:test is empty
            ds[0]

        ds = io.Dataset(dataset_dir, partition_name="funky")
        assert set(ds.list_partitions()) == set(["funky", "default"])
        assert ds.get_partition() == "funky"  # use default normally
        assert set(ds.list_splits()) == set(["train", "valid", "test"])
        assert len(ds) == 3

        ds.set_split("train")
        assert ds.get_split() == "train"
        assert_same_sample(ds[0], sample3)
        assert len(ds) == 1

        ds.set_split("valid")
        assert ds.get_split() == "valid"
        assert_same_sample(ds[0], sample1)
        assert len(ds) == 1

        ds.set_split("test")
        assert ds.get_split() == "test"
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

        with pytest.raises(ValueError):  # raise ValueError because not partition exists
            _ = io.Dataset(dataset_dir)


def custom_band(value, shape=(4, 4), band_name="test_band"):
    data = np.empty(shape)
    data.fill(value)
    if len(shape) == 3 and shape[2] > 1:
        band_info = io.MultiBand(band_name, alt_names=("tb"), spatial_resolution=20, n_bands=shape[2])
    else:
        band_info = io.SpectralBand(band_name, alt_names=("tb"), spatial_resolution=20, wavelength=0.1)
    return io.Band(data, band_info, 10)


def custom_sample(base_value, n_bands=3, name="test_sample"):
    bands = [custom_band(value=base_value + float(i), band_name=f"Band {i}") for i in (100, 200, 300)]
    return io.Sample(bands, base_value, name)


def test_dataset_statistics():
    with tempfile.TemporaryDirectory() as dataset_dir:
        sample1 = custom_sample(base_value=1, name="sample_001")
        sample1.write(dataset_dir)

        sample2 = custom_sample(base_value=2, name="sample_002")
        sample2.write(dataset_dir)

        sample3 = custom_sample(base_value=3, name="sample_003")
        sample3.write(dataset_dir)

        # Default partition, only train
        partition = io.Partition()
        partition.add("train", sample1.sample_name)
        partition.add("train", sample2.sample_name)
        partition.add("train", sample3.sample_name)
        partition.save(directory=dataset_dir, partition_name="default")

        # Compute statistics : this will create all_bandstats.json
        bandstats(dataset_dir, use_splits=False, values_per_image=None, samples=None)

        # Reload dataset with statistics
        ds2 = io.Dataset(dataset_dir)

        statistics = ds2.get_stats(split_wise=False)

        assert set(statistics.keys()) == set(["Band 100", "Band 200", "Band 300", "label"])
        assert np.equal(statistics["Band 100"].min, 101)
        assert np.equal(statistics["Band 100"].max, 103)
        assert np.equal(statistics["Band 100"].median, 102)
        assert np.equal(statistics["Band 100"].mean, 102)
        assert np.equal(statistics["Band 100"].percentile_1, 101)
        assert np.equal(statistics["Band 100"].percentile_99, 103)

        assert np.equal(statistics["Band 200"].min, 201)
        assert np.equal(statistics["Band 200"].max, 203)
        assert np.equal(statistics["Band 200"].median, 202)
        assert np.equal(statistics["Band 200"].mean, 202)
        assert np.equal(statistics["Band 200"].percentile_1, 201)
        assert np.equal(statistics["Band 200"].percentile_99, 203)

        assert np.equal(statistics["label"].min, 1)
        assert np.equal(statistics["label"].max, 3)
        assert np.equal(statistics["label"].median, 2)
        assert np.equal(statistics["label"].mean, 2)

        print("Done")


if __name__ == "__main__":
    test_pack_4d_dense()
    test_pack_4d_multi_band()
    test_write_read()
    test_dataset_partition()
    test_dataset_withnopartition()
    test_dataset_statistics()
