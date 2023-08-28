import tempfile

import numpy as np


import geobench as gb
from make_benchmark.bandstats import produce_band_stats


def custom_band(value, shape=(4, 4), band_name="test_band"):
    data = np.empty(shape)
    data.fill(value)
    if len(shape) == 3 and shape[2] > 1:
        band_info = gb.MultiBand(
            band_name, alt_names=("tb"), spatial_resolution=20, n_bands=shape[2]
        )
    else:
        band_info = gb.SpectralBand(
            band_name, alt_names=("tb"), spatial_resolution=20, wavelength=0.1
        )
    return gb.Band(data, band_info, 10)


def custom_sample(base_value, n_bands=3, name="test_sample"):
    bands = [
        custom_band(value=base_value + float(i), band_name=f"Band {i}") for i in (100, 200, 300)
    ]
    return gb.Sample(bands, base_value, name)


def test_dataset_statistics():
    with tempfile.TemporaryDirectory() as dataset_dir:
        sample1 = custom_sample(base_value=1, name="sample_001")
        sample1.write(dataset_dir)

        sample2 = custom_sample(base_value=2, name="sample_002")
        sample2.write(dataset_dir)

        sample3 = custom_sample(base_value=3, name="sample_003")
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

        # Default partition, only train
        partition = gb.Partition()
        partition.add("train", sample1.sample_name)
        partition.add("train", sample2.sample_name)
        partition.add("train", sample3.sample_name)
        partition.save(directory=dataset_dir, partition_name="default")

        # Compute statistics : this will create all_bandstats.json
        produce_band_stats(
            gb.GeobenchDataset(dataset_dir, band_names=band_names, partition_name="default"),
            values_per_image=None,
            samples=None,
        )

        # Reload dataset with statistics
        ds2 = gb.GeobenchDataset(dataset_dir, band_names=band_names, partition_name="default")

        statistics = ds2.band_stats

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
