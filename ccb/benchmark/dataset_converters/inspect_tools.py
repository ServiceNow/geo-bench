"""Inspect tools."""
import math
from bdb import Breakpoint
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
from warnings import warn

import ipyplot
import numpy as np
import pandas as pd
from ipyleaflet import Map, Marker, Rectangle
from matplotlib import cm
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from rasterio import warp
from rasterio.crs import CRS
from tqdm.auto import tqdm

from ccb import io
from ccb.io import dataset as io_ds
from ccb.io.dataset import Band, CCBDataset, HyperSpectralBands, Sample, SegmentationClasses, compute_dataset_statistics


def compare(a, b, name, src_a, src_b) -> None:
    """Compare two values."""
    if a != b:
        print(f"Consistancy error with {name} between:\n    {src_a}\n  & {src_b}.\n    {str(a)}\n != {str(b)}")


def plot_band_stats(band_values: Dict[str, np.ndarray], n_cols: int = 4, n_hist_bins: int = None) -> None:
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
    images: Union[Sequence[np.ndarray], np.ndarray],
    percentile_max=99.9,
    ensure_3_channels=True,
    per_channel_scaling=False,
) -> np.ndarray:
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
    return np.asarray(new_images)


def extract_images(
    samples: List[Sample],
    band_names: Sequence[str] = ("red", "green", "blue"),
    percentile_max: float = 99.9,
    resample: bool = False,
    fill_value: int = None,
    date_index: int = 0,
) -> Tuple[np.ndarray, Any]:
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
            sample.dates[date_index : date_index + 1], band_names=band_names, resample=resample, fill_value=fill_value
        )
        img_data = img_data[0].astype(np.float32)
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

    images = float_image_to_uint8(np.asarray(images), percentile_max)  # type:ignore
    return images, labels  # type:ignore


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


def hyperspectral_to_rgb(samples: List[Sample], band_name: str, rgb_extract, percentile_max=99.9):
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


def extract_label_as_image(samples, rgb_images=None, opacity=0.3, percentile_max=99.9):
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

    label_images = float_image_to_uint8(images, percentile_max)

    if rgb_images is not None:
        label_images = [
            label_img.squeeze().astype(np.float32) * opacity + rgb_img.astype(np.float32) * (1 - opacity)
            for label_img, rgb_img in zip(label_images, rgb_images)
        ]
        label_images = float_image_to_uint8(label_images)

    return label_images


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
    elif isinstance(label, io.Band):
        label_img = map_class_id_to_color(label.data, label.band_info.n_classes)
        (label_img,) = float_image_to_uint8([label_img])
        return label_img * opacity + (1 - opacity) * image
    else:
        return image


def extract_bands_with_labels(samples, band_groups=None, draw_label=False, label_patch_size=None, date_index=0):
    """Extract bands."""
    if band_groups is None:
        band_groups = [(band_name,) for band_name in samples[0].band_names]
    all_images = []
    band_names = []
    all_labels = []
    unique_band_names = []
    for i, band_group in enumerate(band_groups):

        images, labels = extract_images(samples, band_names=band_group, date_index=date_index)

        if draw_label:
            images = [overlay_label(image, sample.label, label_patch_size) for image, sample in zip(images, samples)]

        all_images.extend(images)
        all_labels.extend(labels)
        group_name = "-".join(band_group)
        unique_band_names.append(group_name)
        band_names.extend((group_name,) * len(images))

    if isinstance(samples[0].label, Band):
        rgb_images, _ = extract_images(samples, band_names=("red", "green", "blue"), date_index=date_index)
        label_images = extract_label_as_image(samples, rgb_images, percentile_max=99)

        all_images.extend(label_images)
        all_labels.extend((None,) * len(label_images))
        band_names.extend(("label",) * len(label_images))
        unique_band_names.append("label")

    return all_images, band_names, all_labels, unique_band_names


def pack_hyperspectral(img: np.ndarray, n_rows: int, n_cols: int):
    """Extract multiple triplet of channels and concatenated them as a grid of images."""
    height, width, n_channels = img.shape
    assert n_rows * n_cols * 3 <= n_channels
    offset = int((n_channels - n_rows * n_cols * 3) / 2)
    img = img[:, :, offset : n_rows * n_cols * 3 + offset]
    img_grid = np.reshape(np.moveaxis(img, -1, 0), (n_rows, n_cols, 3, height, width))
    img_grid = np.moveaxis(img_grid, 2, -1)  # move the channel back to the end
    assert img_grid.shape == (n_rows, n_cols, height, width, 3)
    return img_grid.swapaxes(1, 2).reshape(n_rows * height, n_cols * width, 3)


def extract_bands(samples, band_groups=None, draw_label=False, label_patch_size=None, date_index=0):
    """For backward compatibility."""
    return extract_bands_with_labels(samples, band_groups, draw_label, label_patch_size, date_index)[:2]


def center_coord(band):
    """Find center coordinates."""
    center = np.array(band.data.shape[:2]) / 2.0
    center = transform_to_4326(band.transform, band.crs, center)
    return tuple(center[::-1])


def transform_to_4326(transform, crs, coord):
    """Transform `coord` from band.crs to EPSG4326."""
    coord = transform * coord
    if crs != CRS.from_epsg(4326):
        xs = np.array([coord[0]])
        ys = np.array([coord[1]])
        xs, ys = warp.transform(src_crs=crs, dst_crs=CRS.from_epsg(4326), xs=xs, ys=ys)
        coord = (xs[0], ys[0])
    return coord


def get_rect(band):
    """Obtain a georeferenced rectangle ready to display in ipyleaflet."""
    sw = transform_to_4326(band.transform, band.crs, (0, 0))
    ne = transform_to_4326(band.transform, band.crs, band.data.shape[:2])
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


def load_and_verify_samples(
    dataset_dir, n_samples, n_hist_bins=100, check_integrity=True, split=None, n_value_per_image=1000
):
    """High level function. Loads samples, perform some statistics and plot histograms."""
    dataset = CCBDataset(dataset_dir, split=split)
    samples = list(tqdm(dataset.iter_dataset(n_samples), desc="Loading Samples"))
    if check_integrity:
        io.check_dataset_integrity(dataset, samples=samples)
    band_values, band_stats = compute_dataset_statistics(samples, n_value_per_image=n_value_per_image)
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

    RGB_resolution: Any = [resolution_dict.get(color, None) for color in ("red", "green", "blue")]
    if RGB_resolution[0] == RGB_resolution[1] and RGB_resolution[0] == RGB_resolution[2]:
        RGB_resolution = RGB_resolution[0]

    return {
        "RGB res": RGB_resolution,
        "NIR res": resolution_dict.get("nir", None),
        "# Sentinel2": sentinel2_count,
        "# Sentinel1": sentinel1_count,
        "Elevation res": elevation_resolution,
        "HS res": hs_resolution,
        "# Spectral": spectral_count,
        "# Bands": len(band_info_list),
    }


# Temporary structure mapping task to sensor type (For display)
SENSORS = {
    "forestnet_v1.0": "Landsat",
    "eurosat": "Sentinel-2",
    "brick_kiln_v1.0": "Sentinel-2",
    "so2sat": "Sentinel-2, Sentinel-1",
    "pv4ger_classification": "RGB",
    "geolifeclef-2022": "RGBN, Elevation",
    "bigearthnet": "Sentinel-2",
    "pv4ger_segmentation": "RGB",
    "nz_cattle_segmentation": "RGB",
    "NeonTree_segmentation": "RGB, Hyperspectral (Neon), Elevation (Lidar)",
    "smallholder_cashew": "Sentinel-2",
    "southAfricaCropType": "Sentinel-2",
    "cvpr_chesapeake_landcover": "RGBN",
    "seasonet": "Sentinel-2",
}

DISPLAY_NAMES = {
    "forestnet_v1.0": "m-forestnet",
    "eurosat": "m-eurosat",
    "brick_kiln_v1.0": "m-brick-kiln",
    "so2sat": "m-so2sat",
    "pv4ger_classification": "m-pv4ger",
    "geolifeclef-2022": "m-geolifeclef",
    "bigearthnet": "m-bigearthnet",
    "pv4ger_segmentation": "m-pv4ger-seg",
    "nz_cattle_segmentation": "m-nz-cattle",
    "NeonTree_segmentation": "m-NeonTree",
    "smallholder_cashew": "m-cashew-plant.",
    "southAfricaCropType": "m-SA-crop-type",
    "cvpr_chesapeake_landcover": "m-chesapeake",
    "vit_small_patch16_224": "ViT-S-timm",
    "scratch_vit_small_patch16_224": "ViT-S-Rnd",
    "vit_tiny_patch16_224": "ViT-T-timm",
    "swinv2_tiny_window16_256": "SwinV2-T-timm",
    "convnext_base": "ConvNeXt-B-timm",
    "resnet18": "ResNet18-timm",
    "resnet50": "ResNet50-timm",
    "millionaid_resnet50": "ResNet50-MillionAID",
    "moco_resnet50": "ResNet50-MoCo-S2",
    "moco_resnet50-multi": "ResNet50-MoCo-S2-multi",
    "moco_resnet18": "ResNet18-MoCo-S2",
    "scratch_resnet18": "ResNet18-Rnd",
    "scratch_resnet50": "ResNet50-Rnd",
    "resnet18_Unet": "ResNet18-U-Net-timm",
    "resnet50_Unet": "ResNet50-U-Net-timm",
    "resnet101_Unet": "ResNet101-U-Net-timm",
    "resnet18_DeepLabV3": "ResNet18 DeepLabV3-timm",
    "resnet50_DeepLabV3": "ResNet50 DeepLabV3-timm",
    "resnet101_DeepLabV3": "ResNet101 DeepLabV3-timm",
    "moco_vit_small_patch16_224": "ViT-S-MoCo-S2",
    "moco_vit_small_patch16_224-multi": "ViT-S-MoCo-S2-multi",
    "dino_resnet50": "ResNet50-DINO-S2",
    "dino_resnet50-multi": "ResNet50-DINO-S2-multi",
    "dino_vit_small_patch16_224": "ViT-S-DINO-S2",
    "dino_vit_small_patch16_224-multi": "ViT-S-DINO-S2-multi",
}


def collect_task_info(task, fix_task_shape=False):
    """Collect information for the given task."""
    loss = task.eval_loss

    if isinstance(loss, type):
        loss = loss()
    try:
        dataset = task.get_dataset(split="train")
        partition = dataset.active_partition.partition_dict
        n_train = len(partition["train"])
        n_valid = len(partition["valid"])
        n_test = len(partition["test"])
        n_geoinfo = 0
        for band in dataset[0].bands:
            
            if band.transform != None:
                n_geoinfo += 1
        

    except Exception as e:
        print(e)
        n_train, n_valid, n_test = -1, -1, -1

    n_classes = getattr(task.label_type, "n_classes", -1)

    # shapes = [band.data.shape for band in dataset[0].bands]
    largest_shape = dataset[0].largest_shape()

    if task.patch_size != largest_shape:
        print(f" *WARNING* task.patch_size = {task.patch_size} != dataset[0].largest_shape() = {largest_shape}.")
        if fix_task_shape:
            dataset_dir = io.CCB_DIR / task.benchmark_name / task.dataset_name

            print(f"Overwritint task_info.pkl to {dataset_dir}.")
            task.patch_size = largest_shape
            task.save(dataset_dir, overwrite=True)

    task_dict = {
        "Name": task.dataset_name,
        "Image Size": " x ".join([str(size) for size in task.patch_size]),
        "Loss": str(loss),
        "Label Type": task.label_type.__class__.__name__,
        "# Classes": int(n_classes),
        "# Time Steps": task.n_time_steps,
        "Train Size": n_train,
        "Val Size": n_valid,
        "Test Size": n_test,
        "Sensors": SENSORS.get(task.dataset_name, None),
        "n_geoinfo": n_geoinfo,
    }
    task_dict.update(summarize_band_info(task.bands_info))
    return task_dict, dataset


def collect_benchmark_info(benchmark_name):
    """Collect information for eacth task in the benchmark."""
    data = []
    for task in io.task_iterator(io.CCB_DIR / benchmark_name):
        print(task.dataset_name)

        task_dict, _ = collect_task_info(task)
        data.append(task_dict)
    return data


def benchmark_data_frame(benchmark_name):
    """Format benchmark information into panda data frame."""
    task_dicts = collect_benchmark_info(benchmark_name)
    column_order = (
        "Name",
        "Image Size",
        "Label Type",
        "# Classes",
        "Train Size",
        "Val Size",
        "Test Size",
        "# Time Steps",
        "# Bands",
        "# Sentinel2",
        "RGB res",
        "NIR res",
        "HS res",
        "Elevation res",
        "Sensors",
        "n_geoinfo",
    )
    df = pd.DataFrame.from_records(task_dicts, columns=column_order)
    pd.set_option("max_colwidth", 300)
    return df


def extract_classification_samples(dataset: io.CCBDataset, num_samples=8, rng=np.random):
    """Extract `num_samples` for each class in `dataset`."""
    label_map = dataset.task_specs.get_label_map()
    n_classes = len(label_map)
    n_per_class = np.ceil(num_samples / n_classes)
    samples = []
    for label, names in label_map.items():
        for sample_name in rng.choice(names, size=int(n_per_class), replace=False):
            samples.append(dataset.get_sample(sample_name))
    return samples[:num_samples]


def replace_str(name):
    """Replace some strings to a more display ready version."""
    replace_dict = {
        "Land principally occupied by agriculture, with significant areas of natural vegetation": "Ag. and vegetation",
        "Non-irrigated arable land": "Non-irrigated land",
        "Complex cultivation patterns": "Cultivation patterns",
        "Fruit trees and berry plantations": "Fruit trees and berry",
    }
    for key, val in replace_dict.items():
        if name is not None:
            name = name.replace(key, val)
    return name


def ipyplot_benchmark(benchmark_name, n_samples, img_width=None):
    """Plot samples from every tasks of a given benchmark."""
    for task in io.task_iterator(io.CCB_DIR / benchmark_name):

        print(f"Task: {task.dataset_name}")

        dataset = task.get_dataset(split="train")

        if isinstance(task.label_type, io.label.Classification):
            samples = extract_classification_samples(dataset, n_samples)
        else:
            indexes = np.random.choice(len(dataset), n_samples, replace=False)
            samples = [dataset[idx] for idx in indexes]

        band_groups = [("red", "green", "blue")] + [(band_name,) for band_name in samples[0].band_names]
        images, band_names, labels, tabs_order = extract_bands_with_labels(samples, band_groups)

        if "label" in tabs_order:
            tabs_order.pop(tabs_order.index("label"))
            tabs_order.insert(0, "label")

        if isinstance(task.label_type, io.SegmentationClasses):
            label_names = None
        else:
            label_names = [replace_str(task.label_type.value_to_str(label)) for label in labels]

        for i, image in enumerate(images):
            if image.shape[2] > 3:
                images[i] = pack_hyperspectral(image, 4, 4)

        ipyplot.plot_class_tabs(
            images=images,
            labels=band_names,
            custom_texts=label_names,
            img_width=img_width,
            max_imgs_per_tab=48,
            tabs_order=tabs_order,
        )


def plot_benchmark(benchmark_name, n_samples, save_dir: Path = Path.home() / "figures", fig_size=None):
    """Plot samples of the benchmark using matplotlib for compact visualization."""
    if save_dir is not None:
        save_dir = save_dir / benchmark_name
        save_dir.mkdir(parents=True, exist_ok=True)

    path_list = []

    # cherry picked to avoid images that are not representative
    seed_dict = {
        "forestnet_v1.0": 0,  # 0
        "eurosat": 4,
        "brick_kiln_v1.0": 4,  # 1
        "so2sat": 0,
        "pv4ger_classification": 0,
        "geolifeclef-2022": 0,
        "bigearthnet": 2,
    }
    for task in io.task_iterator(io.CCB_DIR / benchmark_name):

        if task.dataset_name.startswith("geolifeclef"):
            continue

        print(f"Task: {task.dataset_name}")

        dataset = task.get_dataset(split="train")

        rng = np.random.RandomState(seed_dict.get(task.dataset_name, 0))

        if isinstance(task.label_type, io.label.Classification):
            samples = extract_classification_samples(dataset, n_samples, rng=rng)
        else:
            samples = [dataset[i] for i in rng.choice(len(dataset), size=n_samples)]

        if isinstance(task.label_type, io.SegmentationClasses):
            images, band_names, all_labels, unique_band_names = extract_bands_with_labels(samples, [])
            label_names = [None] * len(images)
        else:
            images, labels = extract_images(samples)
            label_names = [replace_str(task.label_type.value_to_str(label)) for label in labels]

        plot_images(images, label_names, DISPLAY_NAMES[task.dataset_name], fig_size=fig_size)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        if save_dir is not None:
            path = save_dir / f"{task.dataset_name}.png"
            plt.savefig(path, bbox_inches="tight")
            path_list.append(path)

    return path_list


def plot_images(images, names, title, fig_size):
    """Plot images using matplotlib for compact visualization."""
    fig, axs = plt.subplots(1, len(images), figsize=fig_size)
    for image, name, ax in zip(images, names, axs):
        if name is not None:
            for sub_name in name.split(" &\n"):
                ax.plot(np.nan, np.nan, ".", color="k", label=sub_name)
        ax.imshow(image)
        ax.axis("off")
        # ax.set_title(name)
        if name is not None:
            ax.legend()
        # ax.text(5, 5, name, bbox={"facecolor": "white", "pad": 10})

    fig.suptitle(title, fontsize=18, y=1.1)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
