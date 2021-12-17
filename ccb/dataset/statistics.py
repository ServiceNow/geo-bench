
from ccb.dataset import io
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from ipyleaflet import Map, Marker, projections, Rectangle


def compare(a, b, name, src_a, src_b):
    if a != b:
        print(f"Consistancy error with {name} between:\n    {src_a}\n  & {src_b}.\n    {str(a)}\n != {str(b)}")


def dataset_statistics(dataset_iterator, n_value_per_images=100, iterator_len=None):

    label_counter = Counter()

    band_stats = defaultdict(list)

    for i, sample in enumerate(dataset_iterator):

        band_count = sample.band_count()
        lenghts = np.array((len(sample.band_names), len(sample.band_wavelength), sample.image.shape[2]))
        if np.any(band_count != lenghts):
            raise ValueError(
                f"Inconsistant number of bands in {sample.name}. Note, the image shape should be (height, width, bands).")

        if i == 0:

            spatial_resolution = sample.spatial_resolution
            band_names = sample.band_names
            crs = sample.crs
            first_name = sample.name

            # TODO:
            #  * band_wavelength
            #  * meta_info

        else:
            compare(spatial_resolution, sample.spatial_resolution, "spatial_resolution", first_name, sample.name)
            compare(tuple(band_names), tuple(sample.band_names), "band_names", first_name, sample.name)
            compare(crs, sample.crs, "crs", first_name, sample.name)

        for idx, band_name in enumerate(band_names):
            band_stats[band_name].append(np.random.choice(sample.get_band(idx).flat,
                                                          size=n_value_per_images, replace=False))

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
    keys, values = zip(*band_values.items())
    fig1, ax = plt.subplots()
    ax.set_title("Band Statistics")
    ax.violinplot(dataset=values, vert=False)
    plt.xlabel("uint16 value")
    ax.set_yticks(np.arange(len(keys)) + 1)
    ax.set_yticklabels(labels=keys)


def extract_images(samples, bands=(3, 2, 1), percentile_max=99.9):
    images = []
    labels = []
    for sample in samples:
        img_data = sample.image[:, :, bands].astype(np.float)
        images.append(img_data)
        labels.append(sample.label)

    mx = np.percentile(images, q=percentile_max)
    new_images = []
    for image in images:
        image = np.clip(image * 255 / mx, 0, 255)
        new_images.append(image.astype(np.uint8))
    return new_images, labels


def extract_bands(samples):
    band_names = samples[0].band_names
    # band_names = [f"{i:02d} {band_name}" for i, band_name in enumerate(band_names)]
    all_images = []
    labels = []
    for i, band_name in enumerate(band_names):
        images, _ = extract_images(samples, i)
        all_images.extend(images)
        labels.extend((band_name, ) * len(images))

    return all_images, labels


def center_coord(sample):
    # TODO why do I have to reverse lon,lat ?
    center = np.array(sample.image.shape[:2]) / 2.
    return tuple((sample.transform * center)[::-1])


def get_rect(sample):
    sw = sample.transform * (0, 0)
    ne = sample.transform * sample.image.shape[:2]
    return Rectangle(bounds=(sw[::-1], ne[::-1]))


def leaflet_map(samples):

    map = Map(center=center_coord(samples[0]), zoom=7)
    map.layout.height = '800px'

    for sample in tqdm(samples):

        map.add_layer(Marker(location=center_coord(sample), draggable=False,
                             opacity=0.5, title=sample.name, alt=sample.name))
        map.add_layer(get_rect(sample))

    return map
