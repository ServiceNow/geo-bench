import rasterio
import numpy as np
import json
import os

# TODO replace by environment variable CC_BENCHMARK_SOURCE_DATASETS
src_datasets_dir = os.path.expanduser("~/dataset/")
dst_datasets_dir = os.path.expanduser("~/converted_dataset/")

SENTINEL2_BAND_NAMES = """\
Band 1 - Coastal aerosol
Band 2 - Blue
Band 3 - Green
Band 4 - Red
Band 5 - Vegetation Red Edge
Band 6 - Vegetation Red Edge
Band 7 - Vegetation Red Edge
Band 8 - NIR
Band 8A - Vegetation Red Edge
Band 9 - Water vapour
Band 10 - SWIR - Cirrus
Band 11 - SWIR
Band 12 - SWIR
""".split("\n")

SENTINEL2_CENTRAL_WAVELENGTHS = """\
0.443
0.49
0.56
0.665
0.705
0.74
0.783
0.842
0.865
0.945
1.375
1.61
2.19""".split("\n")

SENTINEL2_CENTRAL_WAVELENGTHS = [float(wl) for wl in SENTINEL2_CENTRAL_WAVELENGTHS]


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
    resolution: float number. The spatial resolution in meters per pixel.
    band_names: List of strings of len n_bands. Describe the name of each bands in iamge
    band_wavelength: List of float of len n_bands. Central wavelenth for each band in um (micrometers). Use 0 for not a wavelength.
    transform: Affine transformation mapping from pixel coordinates to georeferenced coordinates.
        This should be computed using rasterio.transform using one of the following:
        from_bounds: When the top-left and bottom-right corner of the images are known (no rotation).
        form_gcps: When the coordinates of the 4 corners of the image are required to specify its
            affine transformation (this implies a rotation).
        from_origin: When the top-left corner and pixel size are known.
    crs: Coordinate reference system used for transform. Defaults to 'EPSG:4326', which is
        the most common.
    meta_info: Any extra information that will be stored in tags. Will be serialized using json.dumps.
    """

    def __init__(
            self, image, label, resolution, band_names, band_wavelength, transform, crs='EPSG:4326',
            meta_info=None) -> None:

        # add no data value
        # discuss post-processing
        # discuss perband resolution
        #       Store different resolution in different files (or at least document the resolution)
        # CRS: make sure we use the same CRS used in the dataset we're converting from.
        self.image = image
        self.label = label
        self.resolution = resolution  # rename to spatial resolution
        self.band_names = band_names
        self.band_wavelength = band_wavelength
        self.transform = transform
        self.crs = crs
        self.meta_info = meta_info

    def to_geotiff(self, path):
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

        image = self.image.astype(np.uint16)

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

            data = dict(label=self.label, resolution=self.resolution, meta_info=self.meta_info)
            dst.update_tags(data=json.dumps(data))

            for band_idx in range(image.shape[2]):
                dst.write(image[:, :, band_idx], band_idx + 1)
                if self.band_names is not None:
                    dst.set_band_description(band_idx + 1, self.band_names[band_idx])
                if self.band_wavelength is not None:
                    dst.update_tags(band_idx + 1, wavelength=self.band_wavelength[band_idx])

    def __str__(self):
        str_list = []
        str_list.append("Tiff image with crs: %s and resolution %.3g m/pix." % (self.crs, self.resolution))
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
            image, label=tags["label"],
            resolution=tags["resolution"],
            band_names=list(src.descriptions),
            band_wavelength=wavelenghts,
            transform=src.transform, crs=src.crs,
            meta_info=tags["meta_info"])

    return sample


# def write_to_geotiff(
#         path, image, label, resolution, band_names, band_wavelength, transform, crs='EPSG:4326', meta_info=None):
#     """
#     Write an image from an array to a geotiff file with its label.

#     We compress with zstd, a lossless compression which gains a factor of ~2 in compression.
#     Write speed can be 4x-5x slower and read speed ~2x slower.
#     Interesting benchmark can be found here
#     https://kokoalberti.com/articles/geotiff-compression-optimization-guide/

#     Arguments:
#         path: Destination path to save the file
#         image: array of shape (n_bands, height, width). Will be converted to uint16. ValueError
#             is raised when values are not in range [0, 65535].
#         label: The label of the image. Will be serialized using json.dumps.
#             If the label is a 2d array of shape (height, width), e.g., semantic segmentation, it should
#             be stored as a band, the label should be "band_%d" % band_index, and the band name should
#             be "label".
#         resolution: float number. The spatial resolution in meters per pixel.
#         band_names: List of strings of len n_bands. Describe the name of each bands in iamge
#         band_wavelength: List of float of len n_bands. Central wavelenth for each band. Use 0 for not a wavelength.
#         transform: Affine transformation mapping from pixel coordinates to georeferenced coordinates.
#             This should be computed using rasterio.transform using one of the following:
#             from_bounds: When the top-left and bottom-right corner of the images are known (no rotation).
#             form_gcps: When the coordinates of the 4 corners of the image are required to specify its
#                 affine transformation (this implies a rotation).
#             from_origin: When the top-left corner and pixel size are known.
#         crs: Coordinate reference system used for transform. Defaults to 'EPSG:4326', which is
#             the most common.
#         meta_info: Any extra information that will be stored in tags using json.dumps.
#     Raises:
#         ValueError: when values of image are not in range [0, 65535]
#     """

#     if np.min(image) < 0 or np.max(image) > 65535:
#         raise ValueError("Data out of range. Will not convert to uint16.")
#     image = image.astype(np.uint16)

#     with rasterio.open(path, 'w',
#                        driver='GTiff',
#                        height=image.shape[1],
#                        width=image.shape[2],
#                        count=image.shape[0],
#                        dtype=np.uint16,
#                        crs=crs,
#                        compress="zstd",
#                        predictor=2,
#                        transform=self.transform,
#                        ) as dst:

#         dst.update_tags(label=json.dumps(label))
#         if meta_info is not None:
#             dst.update_tags(meta_info=json.dumps(meta_info))
#         dst.update_tags(resolution=resolution)

#         for band_idx in range(image.shape[0]):
#             dst.write(image[band_idx, :, :], band_idx + 1)
#             dst.set_band_description(band_idx + 1, band_names[band_idx])
#             if band_wavelength is not None:
#                 dst.update_tags(band_idx + 1, wavelength=band_wavelength[band_idx])
