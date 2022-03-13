from ccb import io
import numpy as np
import tempfile


def random_band(shape=(16, 16), band_name="test_band"):
    data = np.random.randint(1, 1000, shape, dtype=np.int16).astype(np.float)
    data *= 2.1
    if len(shape) == 3 and shape[2] > 1:
        band_info = io.MultiBand(band_name, alt_names=("tb"), spatial_resolution=20, n_bands=shape[2])
    else:
        band_info = io.SpectralBand(band_name, alt_names=("tb"), spatial_resolution=20, wavelength=0.1)
    return io.Band(data, band_info, 10)


def random_sample(n_bands=3):
    bands = [random_band(band_name=f"{i:2d}") for i in range(n_bands)]
    return io.Sample(bands, np.random.randint(2), "test_sample")


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
