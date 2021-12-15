
import numpy as np
import rasterio
import json
from pathlib import Path


def swap_band_axes_to_last(image):
    """rearranges axes from (n_bands, height, width) to (height, width, n_bands)."""
    if image.ndim != 3:
        raise ValueError("image must be 3 dimensional array with axes (n_bands, height, width).")
    return image.transpose([1, 2, 0])


class Sample:
    """
    Attributes
    ----------
    image: array of shape (height, width, n_bands). Will be converted to uint16. ValueError
        is raised when values are not in range [0, 65535].
    label: The label of the image. Will be serialized using json.dumps.
        If the label is a 2d array of shape (height, width), e.g., semantic segmentation, it should
        be stored as a band, the label should be "band_%d" % band_index, and the band name should
        be "label".
    spatial_resolution: float number. The spatial resolution in meters per pixel.
    band_names: List of strings of len n_bands. Describe the name of each bands in image
    band_wavelength: List of float of len n_bands. Central wavelenth for each band in um (micrometers). Use 0 for not a wavelength.
    transform: Affine transformation mapping from pixel coordinates to georeferenced coordinates.
        This should be computed using rasterio.transform using one of the following:
        from_bounds: When the top-left and bottom-right corner of the images are known (no rotation).
        form_gcps: When the coordinates of the 4 corners of the image are required to specify its
            affine transformation (this implies a rotation).
        from_origin: When the top-left corner and pixel size are known.
    crs: Coordinate reference system used for transform. Defaults to 'EPSG:4326'. Make sure to provide the same CRS used in
        the original dataset.
    meta_info: Any extra information that will be stored in tags. Will be serialized using json.dumps.
    """

    def __init__(
            self, name, image, label, spatial_resolution, band_names, band_wavelength, transform, crs='EPSG:4326',
            meta_info=None) -> None:

        self.name = name
        self.image = image
        self.label = label
        self.spatial_resolution = spatial_resolution
        self.band_names = band_names
        self.band_wavelength = band_wavelength
        self.transform = transform
        self.crs = crs
        self.meta_info = meta_info

    def to_geotiff(self, directory):
        """
        Write an image from an array to a geotiff file with its label.

        We compress with zstd, a lossless compression which gains a factor of ~2 in compression.
        Write speed can be 4x-5x slower and read speed ~2x slower.
        Interesting benchmark can be found here
        https://kokoalberti.com/articles/geotiff-compression-optimization-guide/

        Arguments:
            path: Destination path to save the file.

        Raises:
            ValueError: when values of image are not in range [0, 65535]
        """
        if np.min(self.image) < 0 or np.max(self.image) > 65535:
            raise ValueError("Data out of range. Will not convert to uint16.")

        if np.sum(np.logical_and(self.image > 1e-6, self.image <= 0.5)) > 0:
            raise ValueError(
                "Float value between 1e-6 and 0.5 would be converted to 0 when casting to uint16, which is the nodata value.")

        image = np.round(self.image).astype(np.uint16)

        path = Path(directory, self.name + ".tif")
        with rasterio.open(path, 'w',
                           driver='GTiff',
                           height=image.shape[0],
                           width=image.shape[1],
                           count=image.shape[2],
                           dtype=np.uint16,
                           crs=self.crs,
                           compress="zstd",
                           predictor=2,
                           transform=self.transform,
                           ) as dst:

            data = dict(label=self.label, spatial_resolution=self.spatial_resolution, meta_info=self.meta_info)
            dst.update_tags(data=json.dumps(data))
            dst.nodata = 0  # we use 0 as the nodata value.
            for band_idx in range(image.shape[2]):
                dst.write(image[:, :, band_idx], band_idx + 1)
                if self.band_names is not None:
                    dst.set_band_description(band_idx + 1, self.band_names[band_idx])
                if self.band_wavelength is not None:
                    dst.update_tags(band_idx + 1, wavelength=self.band_wavelength[band_idx])

    def get_band(self, band_idx):
        return self.image[:, :, band_idx]

    def band_count(self):
        return len(self.band_names)

    def __str__(self):
        str_list = []
        str_list.append("Tiff image with crs: %s and spatial_resolution %.3g m/pix." %
                        (self.crs, self.spatial_resolution))
        str_list.append("Height: %d, width: %d, n_bands: %d" % self.image.shape)
        str_list.append("Label: %s" % self.label)
        str_list.append("Transform:\n%s" % str(self.transform))
        str_list.append("%d bands:" % self.image.shape[2])
        for band_idx in range(self.image.shape[2]):
            band_name = self.band_names[band_idx] if self.band_names is not None else "unnamed"
            wavelngth = self.band_wavelength[band_idx] if self.band_wavelength is not None else 0
            str_list.append("%2d: %6.3fum, %s." % (band_idx + 1, wavelngth, band_name))
        str_list.append("")
        return '\n'.join(str_list)


def from_geotiff(file_path):
    with rasterio.open(file_path) as src:
        tags = json.loads(src.tags()["data"])
        image = swap_band_axes_to_last(src.read())

        wavelenghts = []
        for band_idx in range(image.shape[2]):
            wavelenghts.append(float(src.tags(band_idx + 1)["wavelength"]))

        sample = Sample(
            file_path.stem, image, label=tags["label"],
            spatial_resolution=tags["spatial_resolution"],
            band_names=list(src.descriptions),
            band_wavelength=wavelenghts,
            transform=src.transform, crs=src.crs,
            meta_info=tags["meta_info"])

    return sample


class Partition(dict):

    def __init__(self, partition_dict=None, map=None) -> None:
        self.map = map
        if partition_dict is None:
            self.partition_dict = {'train': [], 'valid': [], 'test': []}
        else:
            self.partition_dict = partition_dict

    def add(self, key, value):
        if key in self.map:
            key = self.map[key]
        self.partition_dict[key].append(value)

    def save(self, file_path):
        with open(file_path, 'w') as fd:
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
            if p.name.endswith('partition.json'):
                self._partition_path_list.append(p)
            elif p.name == "task_specification.json":
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
        for p in path_list:
            yield from_geotiff(p)

    def iter_dataset(self, max_count=None):
        if max_count is None:
            max_count = len(self._sample_path_list)
        return GeneratorWithLength(self._iter_dataset(max_count=max_count), max_count)
