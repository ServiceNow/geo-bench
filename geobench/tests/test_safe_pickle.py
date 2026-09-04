"""Tests for restricted unpickling."""

import datetime
import os
import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio

import geobench as gb
from geobench._safe_pickle import UnsafePickleError, safe_loads


def make_sample(name="test_sample"):
    """Build a sample exercising every metadata type geobench pickles."""
    transform = rasterio.transform.from_bounds(1, 2, 3, 3, 4, 5)
    crs = rasterio.crs.CRS.from_epsg(32631)
    band = gb.Band(
        data=np.random.randint(0, 100, (4, 4)).astype(np.int16),
        band_info=gb.Sentinel2("red", ("B04",), spatial_resolution=10, wavelength=0.665),
        spatial_resolution=np.int64(10),
        date=datetime.date(2020, 1, 1),
        transform=transform,
        crs=crs,
        meta_info={"latitude": 1.5, "longitude": np.float32(2.5)},
    )
    label = gb.Band(
        data=np.random.randint(0, 3, (4, 4)).astype(np.int16),
        band_info=gb.SegmentationClasses("label", spatial_resolution=10, n_classes=3),
        spatial_resolution=10,
        date=datetime.date(2020, 1, 1),
        transform=transform,
        crs=crs,
    )
    return gb.Sample([band], label=label, sample_name=name)


@pytest.mark.parametrize("format", ["hdf5", "npz"])
def test_sample_round_trip(format):
    """Samples written by geobench still load through the restricted unpickler."""
    write, load = {
        "hdf5": (gb.write_sample_hdf5, gb.load_sample_hdf5),
        "npz": (gb.write_sample_npz, gb.load_sample_npz),
    }[format]
    with tempfile.TemporaryDirectory() as dataset_dir:
        sample = make_sample()
        loaded = load(Path(write(sample, dataset_dir)))

        assert loaded.bands[0].band_info.name == "red"
        assert loaded.bands[0].date == datetime.date(2020, 1, 1)
        assert loaded.bands[0].meta_info["latitude"] == 1.5
        assert loaded.bands[0].spatial_resolution == 10
        np.testing.assert_array_equal(loaded.bands[0].data, sample.bands[0].data)


def test_tif_round_trip():
    """Band metadata in GeoTIFF tags still loads through the restricted unpickler."""
    with tempfile.TemporaryDirectory() as dataset_dir:
        sample = make_sample()
        sample_dir = gb.write_sample_tif(sample, dataset_dir)
        band = gb.load_band_tif(Path(sample_dir, "red_2020-01-01.tif"))

        assert band.band_info.name == "red"
        assert band.date == datetime.date(2020, 1, 1)
        assert band.crs == rasterio.crs.CRS.from_epsg(32631)


def test_task_specs_round_trip():
    """Task specifications still load through the restricted unpickler."""
    with tempfile.TemporaryDirectory() as dataset_dir:
        task_specs = gb.TaskSpecifications(
            dataset_name="test",
            bands_info=[gb.Sentinel2("red", spatial_resolution=10)],
            spatial_resolution=np.float64(10.0),
            benchmark_name="test_bench",
            patch_size=(4, 4),
            n_time_steps=1,
            label_type=gb.Classification(3, ["a", "b", "c"]),
        )
        task_specs.save(dataset_dir, overwrite=True)
        loaded = gb.load_task_specs(Path(dataset_dir))

        assert loaded.dataset_name == Path(dataset_dir).name
        assert loaded.label_type.n_classes == 3
        assert loaded.spatial_resolution == 10.0


class _OsSystem:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


class _Eval:
    def __reduce__(self):
        return (eval, ("__import__('os').system('echo pwned')",))


class _Subprocess:
    def __reduce__(self):
        return (subprocess.run, (["echo", "pwned"],))


class _NumpyObjectScalar:
    """The object-dtype gadget: numpy calls pickle.loads to restore the value."""

    def __reduce__(self):
        from numpy._core.multiarray import scalar

        return (scalar, (np.dtype("O"), pickle.dumps(_OsSystem())))


class _ObjectArray:
    """An object-dtype array smuggling a payload past the type allowlist."""

    def __reduce__(self):
        array = np.empty(1, dtype=object)
        array[0] = _OsSystem()
        return (_identity, (array,))


class _NdarraySubclass:
    def __reduce__(self):
        return (_identity, (np.ma.MaskedArray([1, 2, 3]),))


def _identity(value):
    return value


class _ImportedName:
    """A class reachable from a geobench module only because it is imported there."""

    def __reduce__(self):
        return (gb.dataset.Path, ("/tmp/pwned",))


@pytest.mark.parametrize(
    "payload",
    [
        _OsSystem,
        _Eval,
        _Subprocess,
        _NumpyObjectScalar,
        _ObjectArray,
        _NdarraySubclass,
        _ImportedName,
    ],
    ids=lambda cls: cls.__name__,
)
def test_malicious_payload_is_refused(payload):
    """Pickles reaching outside the geobench data format are refused."""
    with pytest.raises(UnsafePickleError):
        safe_loads(pickle.dumps(payload()))


def test_legacy_module_names_are_accepted():
    """Pickles written under the historical `ccb.io` module names still load."""
    try:
        gb.Sentinel2.__module__ = "ccb.io.dataset"
        legacy = pickle.dumps(gb.Sentinel2("red", spatial_resolution=10))
    finally:
        gb.Sentinel2.__module__ = "geobench.dataset"

    assert b"ccb.io.dataset" in legacy
    assert safe_loads(legacy).name == "red"


def test_numeric_arrays_still_load():
    """Numeric arrays are unaffected by the object-dtype restriction."""
    array = np.arange(6).reshape(2, 3)

    np.testing.assert_array_equal(safe_loads(pickle.dumps(array)), array)
