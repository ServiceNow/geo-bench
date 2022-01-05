import ast
import pathlib
import numpy as np
import rasterio
import json
from pathlib import Path
import datetime
from typing import Union
import os
from scipy.ndimage import zoom
import pickle

# TODO replace by environment variable CC_BENCHMARK_SOURCE_DATASETS
src_datasets_dir = os.path.expanduser("~/dataset/")
dst_datasets_dir = os.path.expanduser("~/converted_dataset/")


def _format_date(date: Union[datetime.date, datetime.datetime]):
    if isinstance(date, datetime.date):
        return date.strftime("%Y-%m-%d")
    elif isinstance(date, datetime.datetime):
        return date.strftime("%Y-%m-%d_%H-%M-%S-%Z")
    elif date is None:
        return "NoDate"
    else:
        raise ValueError(f"Unknown date of type: {type(date)}.")


def _date_from_str(date_str):
    if date_str == "NoDate":
        return None
    elif len(date_str) <= 12:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    else:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S-%Z")


class BandInfo(object):
    def __init__(self, name=None, alt_names=(), spatial_resolution=None) -> None:
        self.name = name
        self.alt_names = alt_names
        self.spatial_resolution = spatial_resolution

    def __key(self):
        return self.name

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__key() == other.__key()
        else:
            return False

    def __lt__(self, other):
        return self.__key() < other.__key()


class SpectralBand(BandInfo):
    def __init__(self, name=None, alt_names=(), spatial_resolution=None, wavelength=None) -> None:
        super().__init__(name, alt_names, spatial_resolution)
        self.wavelength = wavelength

    def __key(self):
        return (self.name, self.wavelength)


class Sentinel2(SpectralBand):
    pass


class Mask(BandInfo):
    pass


class CloudProbability(Mask):
    def __init__(self, alt_names=(), spatial_resolution=None) -> None:
        super().__init__("Cloud Probability", alt_names=alt_names, spatial_resolution=spatial_resolution)


class Label(object):
    pass


class Classification(Label):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.n_classes = n_classes


class Regression(Label):
    pass


class SegmentationClasses(BandInfo, Label):
    def __init__(self, name, spatial_resolution, n_classes) -> None:
        super().__init__(name=name, spatial_resolution=spatial_resolution)
        self.n_classes = n_classes


sentinel2_13_bands = [
    Sentinel2("01 - Coastal aerosol", ("1", "01"), 60, 0.443),
    Sentinel2("02 - Blue", ("2", "02", "blue"), 10, 0.49),
    Sentinel2("03 - Green", ("3", "03", "green"), 10, 0.56),
    Sentinel2("04 - Red", ("4", "04", "red"), 10, 0.665),
    Sentinel2("05 - Vegetation Red Edge", ("5", "05"), 20, 0.705),
    Sentinel2("06 - Vegetation Red Edge", ("6", "06"), 20, 0.74),
    Sentinel2("07 - Vegetation Red Edge", ("7", "07"), 20, 0.783),
    Sentinel2("08 - NIR", ("8", "08", "NIR"), 20, 0.842),
    Sentinel2("08A - Vegetation Red Edge", ("8A", "08A"), 20, 0.865),
    Sentinel2("09 - Water vapour", ("9", "09"), 60, 0.945),
    Sentinel2("10 - SWIR - Cirrus", ("10",), 60, 1.375),
    Sentinel2("11 - SWIR", ("11",), 20, 1.61),
    Sentinel2("12 - SWIR", ("12",), 20, 2.19),
]


class Band:
    def __init__(
        self,
        data,
        band_info,
        spatial_resolution,
        date=None,
        transform=None,
        crs=None,
        meta_info=None,
        convert_to_int16=True,
    ) -> None:

        self.data = data
        self.band_info = band_info
        self.spatial_resolution = spatial_resolution
        self.date = date
        self.transform = transform
        self.crs = crs
        self.meta_info = meta_info
        self.convert_to_int16 = convert_to_int16

    def get_descriptor(self):
        descriptor = self.band_info.name
        if self.date is not None:
            descriptor += f"_{_format_date(self.date)}"
        return descriptor

    def to_geotiff(self, directory):
        """
        Write an image from an array to a geotiff file with its label.

        We compress with zstd, a lossless compression which gains a factor of ~2 in compression.
        Write speed can be 4x-5x slower and read speed ~2x slower.
        Interesting benchmark can be found here
        https://kokoalberti.com/articles/geotiff-compression-optimization-guide/

        Arguments:
            directory: Destination path to save the file.

        Raises:
            ValueError: when values of image are not in range (-32768, 32767)
        """
        data = self.data

        if self.convert_to_int16:

            if np.min(data) < -32768 or np.max(data) > 32767:
                raise ValueError("Data out of range. Will not convert to int16.")

            if np.sum(np.logical_and(data > 1e-6, data <= 0.5)) > 0:
                raise ValueError(
                    "Float value between 1e-6 and 0.5 would be converted to 0 when casting to int16, which is the nodata value."
                )

            data = np.round(data).astype(np.int16)

        file_path = Path(directory, f"{self.get_descriptor()}.tif")
        with rasterio.open(
            file_path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=self.crs,
            compress="zstd",
            predictor=2,
            transform=self.transform,
        ) as dst:

            tags = dict(
                date=_format_date(self.date),
                spatial_resolution=self.spatial_resolution,
                band_info=self.band_info,
                meta_info=self.meta_info,
            )
            dst.update_tags(data=str(pickle.dumps(tags)))

            dst.nodata = 0  # we use 0 as the nodata value.
            dst.write(data[:, :], 1)
            dst.set_band_description(1, self.band_info.name)


def load_band_data(file_path):
    with rasterio.open(file_path) as src:
        data = pickle.loads(ast.literal_eval(src.tags()["data"]))

        band_info = data["band_info"]
        image = src.read()
        assert image.shape[0] == 1
        band_data = Band(
            data=image[0],
            band_info=band_info,
            spatial_resolution=data["spatial_resolution"],
            date=_date_from_str(data["date"]),
            transform=src.transform,
            crs=src.crs,
            meta_info=data["meta_info"],
        )

    return band_data


def _make_map(elements):
    elements = list(elements)
    elements.sort()
    element_map = {element: i for i, element in enumerate(elements)}
    return element_map, elements


def _map_bands(band_info_set, band_order):
    if band_order is not None:
        band_info_list = band_order
    else:
        band_info_list = list(band_info_set)
        band_info_list.sort()

    band_name_map = {}
    for band_idx, band_info in enumerate(band_info_list):
        band_name_map[band_info.name] = band_idx
        for alt_name in band_info.alt_names:
            band_name_map[alt_name] = band_idx

    return band_name_map, band_info_list


class Sample(object):
    def __init__(self, bands, label, sample_name, band_order=None) -> None:
        super().__init__()
        self.bands = bands
        self.label = label
        self.band_info_list = band_order  # TODO band_order is currently not saved
        self.sample_name = sample_name
        self._build_index()

    def _build_index(self):

        dates = set()
        band_info_set = set()
        bands = self.bands

        for band in bands:
            dates.add(band.date)
            if self.band_info_list is None:
                band_info_set.add(band.band_info)

        self.date_map, self.dates = _make_map(dates)
        self.band_name_map, self.band_info_list = _map_bands(band_info_set, self.band_info_list)
        self.band_names = [band_info.name for band_info in self.band_info_list]

        self.band_array = np.empty((len(self.dates), len(self.band_info_list)), dtype=np.object)

        for band in bands:
            band_idx = self.band_name_map[band.band_info.name]
            date_idx = self.date_map[band.date]
            self.band_array[date_idx, band_idx] = band

    def get_band_info(self, band_name):
        return self.band_info_list[self.band_name_map[band_name]]

    def is_time_series(self):
        return len(self.dates) > 1

    def pack_to_4d(self, dates=None, band_names=None, resample=False, fill_value=None, resample_order=3):
        band_array, dates, band_names = self.get_band_array(dates, band_names)
        shape, dtype = _largest_shape(band_array)
        data_grid = []
        for i in range(band_array.shape[0]):
            data_list = []
            data_grid.append(data_list)
            for j in range(band_array.shape[1]):
                band = band_array[i, j]
                if band is None:
                    if fill_value is not None:
                        data_list.append(np.zeros(shape, dtype=dtype) + fill_value)
                    else:
                        raise ValueError(f"Missing band {band_names[j]} for date {dates[i]:s}, but fill_vlaue is None.")
                else:
                    if band.data.shape != shape:
                        if resample:
                            zoom_factor = np.array(shape) / np.array(band.data.shape)
                            data_list.append(zoom(band.data, zoom=zoom_factor, order=resample_order))
                        else:
                            raise ValueError(
                                f"Band {band_names[j]} has shape {band.shape:s}, max shape is {shape:s}, but resample is set to False."
                            )
                    else:
                        data_list.append(band.data)
        array = np.moveaxis(np.array(data_grid), 1, 3)
        return array, dates, band_names

    def get_band_array(self, dates=None, band_names=None):
        band_array = self.band_array

        if band_names is not None:
            band_indexes = [self.band_name_map[band_name] for band_name in band_names]
            band_array = band_array[:, band_indexes]
        else:
            band_names = self.band_names

        if dates is not None:
            date_indexes = [self.date_map[date] for date in dates]
            band_array = band_array[date_indexes, :]
        else:
            dates = self.dates

        return band_array, dates, band_names

    def pack_to_3d(self, band_names=None, resample=False, fill_value=None, resample_order=3):
        data_4d, _, band_names = self.pack_to_4d(
            band_names=band_names, resample=resample, fill_value=fill_value, resample_order=resample_order
        )
        assert data_4d.shape[0] == 1
        return data_4d[0], band_names

    def save_sample(self, dataset_dir):

        dst_dir = pathlib.Path(dataset_dir, self.sample_name)
        dst_dir.mkdir(exist_ok=True, parents=True)
        for band in self.bands:
            band.to_geotiff(dst_dir)

        if self.label is not None:
            if isinstance(self.label, Band):
                if not isinstance(self.label.band_info, Label):
                    raise ValueError("The label is of type Band, but its band_info is not instance of Label.")
                self.label.to_geotiff(dst_dir)
            else:
                with open(Path(dst_dir, "label.json"), "w") as fd:
                    json.dump(self.label, fd)


def _largest_shape(band_array):
    shape = [0, 0]
    type_set = set()
    for band in band_array.flat:
        if band is None:
            continue
        shape[0] = max(shape[0], band.data.shape[0])
        shape[1] = max(shape[1], band.data.shape[1])
        type_set.add(band.data.dtype)

    assert len(type_set) == 1
    return tuple(shape), type_set.pop()


def _extract_label(band_list):
    """Extract the label information from the band_list. *Note, the band_list is modified.*"""
    labels = set()
    for idx in range(len(band_list) - 1, -1, -1):  # iterate backward to avoid changing list index when popping
        if isinstance(band_list[idx].band_info, Label):
            labels.add(band_list.pop(idx))

    labels.discard(None)
    if len(labels) != 1:
        raise ValueError(f"Found {len(labels)} label while expecting exactly 1 label.")
    return labels.pop()


def load_sample(sample_dir):
    sample_dir = Path(sample_dir)
    band_list = []
    label = None
    for file in sample_dir.iterdir():
        if file.name == "label.json":
            with open(file, "r") as fd:
                label = json.load(fd)
        else:
            band_list.append(load_band_data(file))

    if label is None:
        label = _extract_label(band_list)
    return Sample(band_list, label, sample_name=sample_dir.name)


class Partition(dict):
    def __init__(self, partition_dict=None, map=None) -> None:
        self.map = map
        if partition_dict is None:
            self.partition_dict = {"train": [], "valid": [], "test": []}
        else:
            self.partition_dict = partition_dict

    def add(self, key, value):
        if key in self.map:
            key = self.map[key]
        self.partition_dict[key].append(value)

    def save(self, directory, partition_name):
        file_path = Path(directory, partition_name + ".json")
        with open(file_path, "w") as fd:
            json.dump(self.partition_dict, fd, indent=2)


class GeneratorWithLength(object):
    def __init__(self, generator, length):
        self.generator = generator
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.generator


class Dataset:
    def __init__(self, dataset_dir) -> None:
        self.dataset_dir = Path(dataset_dir)
        self._load_meta_info()

    def _load_meta_info(self) -> None:
        self._partition_path_list = []
        self._task_specification_path = None
        self._sample_path_list = []
        for p in self.dataset_dir.iterdir():
            if p.name.endswith("partition.json"):
                self._partition_path_list.append(p)
            elif p.name == "task_specifications.pkl":
                self._task_specification_path = p
            else:
                self._sample_path_list.append(p)

    def _iter_dataset(self, max_count=None):
        n = len(self._sample_path_list)
        if max_count is None:
            max_count = n
        else:
            max_count = min(n, max_count)

        path_list = np.random.choice(self._sample_path_list, size=max_count, replace=False)
        for directory in path_list:
            yield load_sample(directory)

    def iter_dataset(self, max_count=None):
        if max_count is None:
            max_count = len(self._sample_path_list)
        return GeneratorWithLength(self._iter_dataset(max_count=max_count), max_count)
