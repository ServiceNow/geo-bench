from ccb.dataset import io
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from ipyleaflet import Map, Marker, projections, Rectangle
import math


def compare(a, b, name, src_a, src_b):
    if a != b:
        print(f"Consistancy error with {name} between:\n    {src_a}\n  & {src_b}.\n    {str(a)}\n != {str(b)}")


def dataset_statistics(dataset_iterator, n_value_per_images=1000, iterator_len=None):

    label_counter = Counter()

    band_stats = defaultdict(list)

    for i, sample in enumerate(tqdm(dataset_iterator)):

        for band in sample.bands:
            band_name = band.band_info.name
            band_stats[band_name].append(np.random.choice(band.data.flat, size=n_value_per_images, replace=False))

        if isinstance(sample.label, io.Band):
            band_stats["label"].append(np.random.choice(sample.label.data.flat, size=n_value_per_images, replace=False))
        else:
            label_counter[sample.label] += 1

    band_values = {}
    for band_name, values in band_stats.items():
        band_values[band_name] = np.hstack(values)

    return band_values, label_counter


def print_stats(label_counter):
    print("Statistics of Labels.")
    for key, count in label_counter.items():
        print(f"{key}: {count}.")


def plot_band_stats(band_values):
    items = list(band_values.items())
    items.sort(key=lambda item: item[0])
    keys, values = zip(*items)
    fig1, ax = plt.subplots()
    ax.set_title("Band Statistics")
    ax.violinplot(dataset=values, vert=False)
    plt.xlabel("uint16 value")
    ax.set_yticks(np.arange(len(keys)) + 1)
    ax.set_yticklabels(labels=keys)


def plot_band_stats2(band_values, n_cols=4):
    items = list(band_values.items())
    items.sort(key=lambda item: item[0])
    n_rows = int(math.ceil(len(items) / n_cols))
    fig1, ax_matrix = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10))
    for i, (key, value) in enumerate(tqdm(items)):
        ax = ax_matrix.flat[i]
        ax.set_title(key)
        ax.hist(value, bins="auto")
    plt.tight_layout()


def float_image_to_uint8(images, percentile_max=99.9):
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
        if not isinstance(sample.label, io.Band):
            raise ValueError("sample.label must be of type Band")

        images.append(sample.label.data)
    return float_image_to_uint8(images, percentile_max)


def extract_bands(samples):
    band_names = samples[0].band_names
    all_images = []
    labels = []
    for i, band_name in enumerate(band_names):
        images, _ = extract_images(samples, band_names=(band_name,))
        images = [image[:, :, 0] for image in images]
        all_images.extend(images)
        labels.extend((band_name,) * len(images))

    if isinstance(samples[0].label, io.Band):
        label_images = extract_label_as_image(samples)
        all_images.extend(label_images)
        labels.extend(("label",) * len(label_images))

    return all_images, labels


def center_coord(band):
    # TODO why do I have to reverse lon,lat ?
    center = np.array(band.data.shape[:2]) / 2.0
    return tuple((band.transform * center)[::-1])


def get_rect(band):
    sw = band.transform * (0, 0)
    ne = band.transform * band.data.shape[:2]
    return Rectangle(bounds=(sw[::-1], ne[::-1]))


def leaflet_map(samples):

    map = Map(center=center_coord(samples[0].bands[0]), zoom=7)
    map.layout.height = "800px"

    for sample in tqdm(samples):
        band = sample.bands[0]
        name = ""
        map.add_layer(Marker(location=center_coord(band), draggable=False, opacity=0.5, title=name, alt=name))
        map.add_layer(get_rect(band))

    return map
