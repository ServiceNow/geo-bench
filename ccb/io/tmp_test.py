from ccb import io
import numpy as np
import tempfile
from ccb.io.dataset_test import random_sample
from ccb.io.bandstats import bandstats

def assert_same_sample(sample, sample_):
    assert sample.sample_name == sample_.sample_name
    assert len(sample.bands) == len(sample_.bands)
    for band in sample.bands:
        len(list(filter(lambda band_: band.band_info == band_.band_info, sample_.bands))) > 0

def test_dataset_partition():
    # Create fake dataset
    with tempfile.TemporaryDirectory() as dataset_dir:
        sample1 = random_sample(name='sample1')
        sample1.write(dataset_dir)

        sample2 = random_sample(name='sample2')
        sample2.write(dataset_dir)

        sample3 = random_sample(name='sample3')
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
        assert set(ds.list_partitions()) == set(['funky', 'default'])
        assert ds.get_partition() == 'default'  # use default normally
        assert set(ds.list_splits()) == set(['train', 'valid', 'test'])
        assert ds.get_split() is None
        assert len(ds) == 3
        # Ordering is not guaranteed. Do we want to enforce that? The following can fail
        #assert_same_sample(ds[0], sample1)
        #assert_same_sample(ds[1], sample2)
        #assert_same_sample(ds[2], sample3)

        ds.set_split('train')
        assert ds.get_split() == 'train'
        assert_same_sample(ds[0], sample1)
        assert len(ds) == 1

        ds.set_split('valid')
        assert ds.get_split() == 'valid'
        # Try strict ordering
        try:
            assert_same_sample(ds[0], sample2)
            assert_same_sample(ds[1], sample3)
        except:
            assert_same_sample(ds[0], sample3)
            assert_same_sample(ds[1], sample2)            
        assert len(ds) == 2

        ds.set_split('test')
        assert ds.get_split() == 'test'
        assert len(ds) == 0
        try:
            ds[0]
            raise Exception('Should have broken here. default:test is empty')
        except IndexError:
            pass # This is correctt

        ds = io.Dataset(dataset_dir, partition_name='funky')
        assert set(ds.list_partitions()) == set(['funky', 'default'])
        assert ds.get_partition() == 'funky'  # use default normally
        assert set(ds.list_splits()) == set(['train', 'valid', 'test'])
        assert len(ds)==3


        ds.set_split('train')
        assert ds.get_split() == 'train'
        assert_same_sample(ds[0], sample3)
        assert len(ds) == 1

        ds.set_split('valid')
        assert ds.get_split() == 'valid'
        assert_same_sample(ds[0], sample1)
        assert len(ds) == 1

        ds.set_split('test')
        assert ds.get_split() == 'test'
        assert_same_sample(ds[0], sample2)
        assert len(ds) == 1
        try:
            ds[2]
            raise Exception('Should have broken here. default:test is empty')
        except IndexError:
            pass # This is correctt


def test_dataset_withnopartition():
    with tempfile.TemporaryDirectory() as dataset_dir:
        sample1 = random_sample(name='sample1')
        sample1.write(dataset_dir)

        sample2 = random_sample(name='sample2')
        sample2.write(dataset_dir)

        sample3 = random_sample(name='sample3')
        sample3.write(dataset_dir)  

        try:
            ds = io.Dataset(dataset_dir)
            raise Exception('This should fail because there is no partition')
        except ValueError:  # raised internally by set_partition()
            pass#ed the test


def custom_band(value, shape=(4, 4), band_name="test_band"):
    data = np.empty(shape)
    data.fill(value)
    if len(shape) == 3 and shape[2] > 1:
        band_info = io.MultiBand(band_name, alt_names=("tb"), spatial_resolution=20, n_bands=shape[2])
    else:
        band_info = io.SpectralBand(band_name, alt_names=("tb"), spatial_resolution=20, wavelength=0.1)
    return io.Band(data, band_info, 10)


def custom_sample(base_value, n_bands=3, name="test_sample"):
    bands = [custom_band(value=base_value + float(i), band_name=f"Band {i}") for i in (100,200,300)]
    return io.Sample(bands, base_value, name)



def test_dataset_statistics():
    with tempfile.TemporaryDirectory() as dataset_dir:
        sample1 = custom_sample(base_value=1, name='sample_001')
        sample1.write(dataset_dir)

        sample2 = custom_sample(base_value=2, name='sample_002')
        sample2.write(dataset_dir)

        sample3 = custom_sample(base_value=3, name='sample_003')
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

        assert set(statistics.keys()) == set(['Band 100','Band 200','Band 300','label'])
        assert np.equal(statistics['Band 100'].min, 101)
        assert np.equal(statistics['Band 100'].max, 103)
        assert np.equal(statistics['Band 100'].median, 102)
        assert np.equal(statistics['Band 100'].mean, 102)
        assert np.equal(statistics['Band 100'].percentile_1, 101)
        assert np.equal(statistics['Band 100'].percentile_99, 103)

        assert np.equal(statistics['Band 200'].min, 201)
        assert np.equal(statistics['Band 200'].max, 203)
        assert np.equal(statistics['Band 200'].median, 202)
        assert np.equal(statistics['Band 200'].mean, 202)
        assert np.equal(statistics['Band 200'].percentile_1, 201)
        assert np.equal(statistics['Band 200'].percentile_99, 203)

        assert np.equal(statistics['label'].min, 1)
        assert np.equal(statistics['label'].max, 3)
        assert np.equal(statistics['label'].median, 2)
        assert np.equal(statistics['label'].mean, 2)

        print('Done')


if __name__ == '__main__':
    test_dataset_partition()
    test_dataset_withnopartition()
    test_dataset_statistics()