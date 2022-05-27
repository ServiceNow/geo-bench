from pathlib import Path
from tempfile import mkdtemp
import time
import numpy as np
from ccb import io
import os


def random_sentinel2_sample(sample_name, max_shape):
    bands = []

    for band_info in io.sentinel2_13_bands:

        shape = int(max_shape // (band_info.spatial_resolution / 10))
        data = (np.random.rand(shape, shape, 1) * 255).astype(np.int16)

        band_data = io.Band(
            data=data,
            band_info=band_info,
            spatial_resolution=band_info.spatial_resolution,
        )
        bands.append(band_data)

    return io.Sample(bands, label=int(np.random.randint(10)), sample_name=sample_name)


def geotiff_writer(sample: io.Sample, dataset_dir):
    sample.write(dataset_dir)
    return dataset_dir / sample.sample_name


def test_read_write_speed(writer, reader, n=100, max_shape=96):

    print(f"Test writer {writer.__name__}, reader {reader.__name__} with max_shape {max_shape}.")
    samples = [random_sentinel2_sample(f"sample_{i:02d}", max_shape=max_shape) for i in range(n)]

    dataset_dir = Path(mkdtemp())
    print("writing to:", dataset_dir)
    t0 = time.time()

    sample_paths = []

    for i, sample in enumerate(samples):
        sample_paths.append(writer(sample, dataset_dir))

    t1 = time.time()

    sizes = 0
    for sample_path in sample_paths:
        # sizes += os.path.getsize(sample_path)
        sample = reader(sample_path)

    t2 = time.time()

    print(f"wrting time: {t1-t0:.2f}.")
    print(f"reading time: {t2-t1:.2f}.")
    print(f"average size: {sizes/len(sample_paths)/1e6:.2f} MB.")


if __name__ == "__main__":

    # test_read_write_speed(io.write_sample_npz, io.load_sample_npz, max_shape=384)

    # test_read_write_speed(io.write_sample_hdf5, io.load_sample_hdf5, max_shape=384)

    test_read_write_speed(io.write_sample_tif, io.load_sample_tif, max_shape=384)
