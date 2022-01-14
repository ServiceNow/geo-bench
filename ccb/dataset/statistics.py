from toolbox.core.task_specs import TaskSpecifications
from ccb.dataset import io
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from ipyleaflet import Map, Marker, projections, Rectangle
import math
from matplotlib import cm
from typing import List
from warnings import warn
from rasterio.crs import CRS
from rasterio import warp


def compare(a, b, name, src_a, src_b):
    if a != b:
        print(f"Consistancy error with {name} between:\n    {src_a}\n  & {src_b}.\n    {str(a)}\n != {str(b)}")


def dataset_statistics(dataset_iterator, n_value_per_image=1000):

    accumulator = defaultdict(list)

    for i, sample in enumerate(tqdm(dataset_iterator, desc="Extracting Statistics")):

        for band in sample.bands:
            accumulator[band.band_info.name].append(
                np.random.choice(band.data.flat, size=n_value_per_image, replace=False)
            )

        if isinstance(sample.label, io.Band):
            accumulator["label"].append(np.random.choice(sample.label.data.flat, size=n_value_per_image, replace=False))
        elif isinstance(sample.label, (list, tuple)):
            for obj in sample.label:
                if isinstance(obj, dict):
                    for key, val in obj.items():
                        accumulator[f"label_{key}"].append(val)
        else:
            accumulator["label"].append(sample.label)

    band_values = {}
    band_stats = {}
    for name, values in accumulator.items():
        values = np.hstack(values)
        band_values[name] = values
        band_stats[name] = io.compute_stats(values)

    return band_values, band_stats


def print_stats(label_counter):
    print("Statistics of Labels.")
    for key, count in label_counter.items():
        print(f"{key}: {count}.")


# def plot_band_stats_violin(band_values):
#     items = list(band_values.items())
#     items.sort(key=lambda item: item[0])
#     keys, values = zip(*items)
#     fig1, ax = plt.subplots()
#     ax.set_title("Band Statistics")
#     ax.violinplot(dataset=values, vert=False)
#     plt.xlabel("uint16 value")
#     ax.set_yticks(np.arange(len(keys)) + 1)
#     ax.set_yticklabels(labels=keys)


def plot_band_stats(band_values, n_cols=4, n_hist_bins=None):
    """Plot a histogram of band values for each band.

    Args:
        band_values: dict of 1d arryay representing flattenned values for each band.
        n_cols: number of columns in the histogram gird
        n_hist_bins: number of bins to use for histograms. See pyplot.hist's bins argument for more details
    """
    items = list(band_values.items())
    items.sort(key=lambda item: item[0])
    n_rows = int(math.ceil(len(items) / n_cols))
    fig1, ax_matrix = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10))
    for i, (key, value) in enumerate(tqdm(items, desc="Plotting statistics")):
        ax = ax_matrix.flat[i]
        ax.set_title(key)
        ax.hist(value, bins=n_hist_bins)
    plt.tight_layout()


def float_image_to_uint8(images, percentile_max=99.9, ensure_3_channels=True):
    """Convert a batch of images to uint8 such that 99.9% of values fit in the range (0,255)."""
    images = np.asarray(images)
    if images.dtype == np.uint8:
        return images

    if np.any(images < 0):
        raise ValueError("Images contain negative values. Can't conver to uint8")

    images = images.astype(np.float64)

    mx = np.percentile(images, q=percentile_max)
    new_images = []
    for image in images:
        image = np.clip(image * (255 / mx), 0, 255)
        if ensure_3_channels:
            if image.ndim == 2:
                image = np.stack((image, image, image), axis=2)
            if image.shape[2] == 1:
                image = np.concatenate((image, image, image), axis=2)
        new_images.append(image.astype(np.uint8))
    return new_images


def extract_images(samples, band_names=("red", "green", "blue"), percentile_max=99.9, resample=False, fill_value=None):
    images = []
    labels = []
    for sample in samples:
        img_data, _, _ = sample.pack_to_4d(sample.dates[:1], band_names, resample=resample, fill_value=fill_value)
        img_data = img_data[0].astype(np.float)
        images.append(img_data)
        labels.append(sample.label)

    images = float_image_to_uint8(images, percentile_max)
    return images, labels


def extract_label_as_image(samples, percentile_max=99.9):
    images = []
    for sample in samples:
        label = sample.label
        if not isinstance(label, io.Band):
            raise ValueError("sample.label must be of type Band")

        if isinstance(label.band_info, io.SegmentationClasses):
            image = map_class_id_to_color(label.data, label.band_info.n_classes)
        else:
            image = label.data
        images.append(image)

    return float_image_to_uint8(images, percentile_max)


def extract_bands(samples, band_groups=None):
    if band_groups is None:
        band_groups = [(band_name,) for band_name in samples[0].band_names]
    all_images = []
    labels = []
    for i, band_group in enumerate(band_groups):
        images, _ = extract_images(samples, band_names=band_group)
        # images = [image[:, :, 0] for image in images]
        all_images.extend(images)
        group_name = '-'.join(band_group)
        labels.extend((group_name,) * len(images))

    if isinstance(samples[0].label, io.Band):
        label_images = extract_label_as_image(samples)
        all_images.extend(label_images)
        labels.extend(("label",) * len(label_images))

    return all_images, labels


def center_coord(band):
    # TODO why do I have to reverse lon,lat ?
    center = np.array(band.data.shape[:2]) / 2.0
    center = transform_to_4326(band, center)
    return tuple(center[::-1])


def transform_to_4326(band, coord):
    coord = band.transform * coord
    if band.crs != CRS.from_epsg(4326):
        xs = np.array([coord[0]])
        ys = np.array([coord[1]])
        xs, ys = warp.transform(src_crs=band.crs, dst_crs=CRS.from_epsg(4326), xs=xs, ys=ys)
        coord = (xs[0], ys[0])
    return coord


def get_rect(band):
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


def load_and_veryify_samples(dataset_dir, n_samples, n_hist_bins=100):
    """High level function. Loads samples, perform some statistics and plot histograms."""
    dataset = io.Dataset(dataset_dir)
    samples = list(tqdm(dataset.iter_dataset(n_samples), desc="Loading Samples"))
    band_values, band_stats = dataset_statistics(samples, n_value_per_image=1000)
    plot_band_stats(band_values=band_values, n_hist_bins=n_hist_bins)
    return dataset, samples, band_values, band_stats


def map_class_id_to_color(id_array, n_classes, background_id=0, background_color=(0, 0, 0)):
    colors = cm.hsv(np.linspace(0, 1, n_classes + 1))
    colors = colors[:, :-1]  # drop the last column since it corresponds to alpha channel.
    colors = colors[:-1]  # drop the last color since it's almost the same as the 1st color.
    colors[background_id, :] = background_color
    image = np.array([map[id_array] for map in colors.T])
    return np.moveaxis(image, 0, 2)


def check_integrity(samples: List[io.Sample], task_specs: TaskSpecifications, assert_dense=True):
    for sample in samples:
        assert len(task_specs.bands_info) == len(sample.band_info_list)
        assert task_specs.n_time_steps == len(sample.dates), f"{task_specs.n_time_steps} vs {len(sample.dates)}"

        shapes = []
        for band in sample.bands:
            band.band_info.assert_valid(band)
            shapes.append(band.data.shape)
        max_shape = np.array(shapes).max(axis=0)
        assert np.all(max_shape == task_specs.patch_size), f"{max_shape} vs {task_specs.patch_size}"

        assert isinstance(task_specs.label_type, io.Label)
        task_specs.label_type.assert_valid(sample.label)

        if assert_dense:
            assert np.all(sample.band_array != None)
