"""Inspect tools."""
import math
from typing import Any, Callable, Dict, List, Sequence, Tuple
from warnings import warn

import ipyplot
import numpy as np
from ccb import io
from ccb.io import dataset as io_ds
from ccb.io.dataset import (Band, Dataset, HyperSpectralBands, Sample,
                            SegmentationClasses, compute_dataset_statistics)
from ipyleaflet import Map, Marker, Rectangle
from matplotlib import cm
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from rasterio import warp
from rasterio.crs import CRS
from tqdm import tqdm


def compare(a, b, name, src_a, src_b) -> None:
    """Compare two values."""
    if a != b:
        print(f"Consistancy error with {name} between:\n    {src_a}\n  & {src_b}.\n    {str(a)}\n != {str(b)}")


def plot_band_stats(band_values: Dict[str, np.array], n_cols: int = 4, n_hist_bins: int = None) -> None:
    """Plot a histogram of band values for each band.

    Args:
        band_values: dict of 1d arryay representing flattenned values for each band.
        n_cols: number of columns in the histogram gird
        n_hist_bins: number of bins to use for histograms. See pyplot.hist's bins argument for more details
    """
    items = list(band_values.items())
    items.sort(key=lambda item: item[0])
    n_rows = int(math.ceil(len(items) / n_cols))
    fig1, ax_matrix = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 5))
    for i, (key, value) in enumerate(tqdm(items, desc="Plotting statistics")):
        ax = ax_matrix.flat[i]
        ax.set_title(key)
        ax.hist(value, bins=n_hist_bins)
    plt.tight_layout()


def float_image_to_uint8(
    images: np.array, percentile_max=99.9, ensure_3_channels=True, per_channel_scaling=False
) -> np.array:
    """Convert a batch of images to uint8 such that 99.9% of values fit in the range (0,255).

    Args:
        images: batch of images
        percentile_max: maximum percentile value
        ensure_3_channels: whether or not to return 3 channel dimensions
        per_channel_scaling: whether or not to apply the scaling per channel

    Returns:
        converted batch of images
    """
    images = np.asarray(images)
    if images.dtype == np.uint8:
        return images

    images = images.astype(np.float64)

    if per_channel_scaling:
        mx = np.percentile(images, q=percentile_max, axis=(0, 1, 2), keepdims=True)
        mx = np.squeeze(mx, axis=0)
        mn = np.percentile(images, q=100 - percentile_max, axis=(0, 1, 2), keepdims=True)
    else:
        mn = np.percentile(images, q=100 - percentile_max)
        mx = np.percentile(images, q=percentile_max)

    new_images = []
    for image in images:
        image = np.clip((image - mn) * 255 / (mx - mn), 0, 255)
        if ensure_3_channels:
            if image.ndim == 2:
                image = np.stack((image, image, image), axis=2)
            if image.shape[2] == 1:
                image = np.concatenate((image, image, image), axis=2)
        new_images.append(image.astype(np.uint8))
    return new_images


def extract_images(
    samples: List[Sample],
    band_names: Sequence[str] = ("red", "green", "blue"),
    percentile_max: float = 99.9,
    resample: bool = False,
    fill_value: int = None,
    date_index: int = 0,
) -> Tuple[List[np.array]]:
    """Extract images from samples.

    Args:
        samples: set of samples
        band_names: band names to extract from sample
        percentile_max: maximum percentile value
        resample: whether or not to resample
        fill_value: fill values
        date_index: for timeseries which date to index

    Returns:
        images and labels extracted from sample
    """
    images = []
    labels = []
    for sample in samples:
        img_data, _, _ = sample.pack_to_4d(
            sample.dates[date_index : date_index + 1], band_names, resample=resample, fill_value=fill_value
        )
        img_data = img_data[0].astype(np.float)
        # TODO We should pass labelType from task specs and compare that instead of the class
        # Once we change this function, we should update all inspection notebooks
        # if isinstance(sample.label, np.ndarray):
        #     for i, label in enumerate(sample.label):
        #         if label == 1:
        #             images.append(img_data)
        #             labels.append(i)
        # else:
        images.append(img_data)
        labels.append(sample.label)

    images = float_image_to_uint8(images, percentile_max)
    return images, labels


def callback_hyperspectral_to_rgb(
    samples: List[Sample], band_name: str, percentile_max: float = 99.9, img_width: int = 128
) -> Callable[[int, int], Any]:
    """Create callable to convert hyperspectral to rgb for plotting.

    Args:
        samples: set of samples

    """

    def callback(center, width):
        rgb_extractor = make_rgb_extractor(center, width)
        images = hyperspectral_to_rgb(samples, band_name, rgb_extractor, percentile_max)
        return ipyplot.plot_images(images=images, img_width=img_width, max_images=len(samples))

    return callback


def make_rgb_extractor(center, width):
    """Create callable to extract rgb data from hyperspectral.

    Args:
        center:
        width:

    Returns:
        callable
    """

    def callback(hs_data):
        def _extrac_band(start, stop):
            return hs_data[:, :, int(start) : int(stop)].mean(axis=2)

        h, w, d = hs_data.shape
        _center = max(0, center - width * 1.5) + width * 1.5
        _center = min(d, _center + width * 1.5) - width * 1.5

        red = _extrac_band(_center - width * 1.5, _center - width * 0.5)
        green = _extrac_band(_center - width * 0.5, _center + width * 0.5)
        blue = _extrac_band(_center + width * 0.5, _center + width * 1.5)

        return np.dstack((red, green, blue))

    return callback


def hyperspectral_to_rgb(samples: List[Sample], band_name, rgb_extract, percentile_max=99.9):
    """Convert hyperspectral to rgb."""
    images = []
    for sample in samples:
        band_array, _, _ = sample.get_band_array(band_names=(band_name,))
        assert band_array.shape == (1, 1), f"Got shape: {band_array.shape}."
        band = band_array[0, 0]
        assert isinstance(band.band_info, HyperSpectralBands), f"Got type: {type(band.band_info)}."
        hs_data = band.data
        images.append(rgb_extract(hs_data))

    return float_image_to_uint8(images, percentile_max, per_channel_scaling=True)


def extract_label_as_image(samples, percentile_max=99.9):
    """If label is a band, will convert into an image. Otherwise, will raise an error."""
    images = []
    for sample in samples:
        label = sample.label
        if not isinstance(label, Band):
            raise ValueError("sample.label must be of type Band")

        if isinstance(label.band_info, SegmentationClasses):
            image = map_class_id_to_color(label.data, label.band_info.n_classes)
        else:
            image = label.data
        images.append(image)

    return float_image_to_uint8(images, percentile_max)


def overlay_label(image, label, label_patch_size, opacity=0.5):
    """Overlay label on image."""
    if label_patch_size is not None:
        scale = np.array(image.shape[:2]) / np.array(label_patch_size)
    else:
        scale = np.array([1.0, 1.0])
    if isinstance(label, (list, tuple)):  # TODO hack tha needs to change
        im = Image.fromarray(image)
        ctxt = ImageDraw.Draw(im)
        for obj in label:
            if isinstance(obj, dict) and "xmin" in obj:
                coord = np.array([[obj["xmin"], obj["ymin"]], [obj["xmax"], obj["ymax"]]])
                ctxt.rectangle(list((coord * scale).flat), outline=(255, 0, 0))
            elif isinstance(obj, (tuple, list)) and len(obj) == 2:
                size = 5 * scale
                coord = [obj[0] - size[0], obj[1] - size[1], obj[0] + size[1], obj[1] + size[1]]
                ctxt.rectangle(coord, outline=(255, 0, 0))
        return np.array(im) * opacity + (1 - opacity) * image
    else:
        return image


def extract_bands(samples, band_groups=None, draw_label=False, label_patch_size=None, date_index=0):
    """Extract bands."""
    if band_groups is None:
        band_groups = [(band_name,) for band_name in samples[0].band_names]
    all_images = []
    labels = []
    for i, band_group in enumerate(band_groups):
        images, _ = extract_images(samples, band_names=band_group, date_index=date_index)
        if draw_label:
            images = [overlay_label(image, sample.label, label_patch_size) for image, sample in zip(images, samples)]

        all_images.extend(images)
        group_name = "-".join(band_group)
        labels.extend((group_name,) * len(images))

    if isinstance(samples[0].label, Band):
        label_images = extract_label_as_image(samples)
        all_images.extend(label_images)
        labels.extend(("label",) * len(label_images))

    return all_images, labels


def center_coord(band):
    """Find center coordinates."""
    center = np.array(band.data.shape[:2]) / 2.0
    center = transform_to_4326(band, center)
    return tuple(center[::-1])


def transform_to_4326(band, coord):
    """Transform `coord` from band.crs to EPSG4326."""
    coord = band.transform * coord
    if band.crs != CRS.from_epsg(4326):
        xs = np.array([coord[0]])
        ys = np.array([coord[1]])
        xs, ys = warp.transform(src_crs=band.crs, dst_crs=CRS.from_epsg(4326), xs=xs, ys=ys)
        coord = (xs[0], ys[0])
    return coord


def get_rect(band):
    """Obtain a georeferenced rectangle ready to display in ipyleaflet."""
    sw = transform_to_4326(band, (0, 0))
    ne = transform_to_4326(band, band.data.shape[:2])
    return Rectangle(bounds=(sw[::-1], ne[::-1]))


def leaflet_map(samples):
    """Position all samples on a world map using ipyleaflet. Experimental feature."""
    # TODO need to use reproject to increse compatibility
    # https://github.com/jupyter-widgets/ipyleaflet/blob/master/examples/Numpy.ipynb

    map = Map(center=center_coord(samples[0].bands[0]), zoom=7)
    map.layout.height = "800px"

    for sample in tqdm(samples):
        band = sample.bands[0]
        if band.crs is None or band.transform is None:
            warn("Unknown transformation or crs.")
            continue
        name = sample.sample_name
        map.add_layer(Marker(location=center_coord(band), draggable=False, opacity=0.5, title=name, alt=name))
        map.add_layer(get_rect(band))

    return map


def load_and_verify_samples(dataset_dir, n_samples, n_hist_bins=100, check_integrity=True, split=None):
    """High level function. Loads samples, perform some statistics and plot histograms."""
    dataset = Dataset(dataset_dir, split=split)
    samples = list(tqdm(dataset.iter_dataset(n_samples), desc="Loading Samples"))
    if check_integrity:
        io.check_dataset_integrity(dataset, samples=samples)
    band_values, band_stats = compute_dataset_statistics(samples, n_value_per_image=1000)
    plot_band_stats(band_values=band_values, n_hist_bins=n_hist_bins)
    return dataset, samples, band_values, band_stats


load_and_veryify_samples = load_and_verify_samples  # compatibility


def map_class_id_to_color(id_array, n_classes, background_id=0, background_color=(0, 0, 0)):
    """Attribute a color for each classes using a rainbow colormap."""
    colors = cm.hsv(np.linspace(0, 1, n_classes + 1))
    colors = colors[:, :-1]  # drop the last column since it corresponds to alpha channel.
    colors = colors[:-1]  # drop the last color since it's almost the same as the 1st color.
    colors[background_id, :] = background_color
    image = np.array([map[id_array] for map in colors.T])
    return np.moveaxis(image, 0, 2)


def summarize_band_info(band_info_list: List[io.BandInfo]):
    """Summarize band info."""
    sentinel2_count = 0
    sentinel1_count = 0
    spectral_count = 0
    elevation_resolution = None
    hs_resolution = None

    resolution_dict = {}

    for band_info in band_info_list:
        if isinstance(band_info, io_ds.SpectralBand):
            spectral_count += 1
        if isinstance(band_info, io_ds.Sentinel1):
            sentinel1_count += 1
        if isinstance(band_info, io_ds.Sentinel2):
            sentinel2_count += 1
        if isinstance(band_info, io_ds.ElevationBand):
            elevation_resolution = band_info.spatial_resolution
        if isinstance(band_info, io_ds.HyperSpectralBands):
            hs_resolution = band_info.spatial_resolution

        resolution_dict[band_info.name.lower()] = band_info.spatial_resolution
        for name in band_info.alt_names:
            resolution_dict[name.lower()] = band_info.spatial_resolution

    RGB_resolution = [resolution_dict.get(color, None) for color in ("red", "green", "blue")]
    if RGB_resolution[0] == RGB_resolution[1] and RGB_resolution[0] == RGB_resolution[2]:
        RGB_resolution = RGB_resolution[0]

    return {
        "RGB res": RGB_resolution,
        "NIR res": resolution_dict.get("nir", None),
        "Sentinel2 count": sentinel2_count,
        "Sentinel1 count": sentinel1_count,
        "Elevation res": elevation_resolution,
        "HS res": hs_resolution,
        "Spectral count": spectral_count,
        "Bands count": len(band_info_list),
    }
