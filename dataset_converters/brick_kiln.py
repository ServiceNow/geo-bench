# Downloaded from "https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg13/brick_kiln.html"

# Questions to authors:
# * what are indices in the csv file
# * how is the train valid test partition selected?
# * why are there blocks of shape 0?
# * confirm that the crs is EPSG:4326

from ccb.dataset import io
import numpy as np
import csv
from dataset_converters import util
import rasterio
from pathlib import Path
import h5py
from tqdm import tqdm


src_dataset_dir = Path(util.src_datasets_dir, "brick_kiln_v1.0")
dataset_dir = Path(util.dst_datasets_dir, "brick_kiln_v1.0")
dataset_dir.mkdir(exist_ok=True, parents=True)


def load_examples_bloc(file_path):
    """Load a .h5py bloc of images with their labels."""
    file_id = file_path.stem.split("_")[1]

    with h5py.File(file_path) as data:

        images = data["images"][:]
        labels = data["labels"][:]
        bounds = data["bounds"][:]

        return images, labels, bounds, file_id


def read_list_eval_partition(csv_file):
    with open(csv_file) as fd:
        reader = csv.reader(fd, delimiter=",")
        data = []
        for i, row in enumerate(reader):
            if i == 0:
                continue
            data.append([float(e) for e in row])

    data = np.array(data)

    y = data[:, 0].astype(np.int)
    partition = data[:, 1].astype(np.int)
    hdf5_file = data[:, 2].astype(np.int)
    hdf5_idx = data[:, 3].astype(np.int)
    gps = data[:, 4:8]
    indices = data[:, 8:11].astype(np.int)

    id_map = {}
    for i, (file_id, sample_idx) in enumerate(zip(hdf5_file, hdf5_idx)):
        id_map[(file_id, sample_idx)] = i

    return y, partition, hdf5_file, hdf5_idx, gps, indices, id_map


# y, partition, hdf5_file, hdf5_idx, gps, indices, id_map = read_list_eval_partition(
#     Path(src_dataset_dir, 'list_eval_partition.csv'))


if __name__ == "__main__":

    _, partition_id, _, _, _, _, id_map = read_list_eval_partition(Path(src_dataset_dir, "list_eval_partition.csv"))

    file_list = list(src_dataset_dir.iterdir())
    partition = io.Partition(map={0: "train", 1: "valid", 2: "test"})

    pbar = tqdm(file_list)
    for file_idx, file_path in enumerate(pbar):
        if file_path.suffix != ".hdf5":
            continue

        images, labels, bounds, file_id = load_examples_bloc(file_path)
        msg = f"Processing {file_path.name}, shape = {str(images.shape)}."
        pbar.set_description(msg)

        if images.shape[0] == 0:
            print("Skipping block of shape 0. Shape = %s" % (str(images.shape)))
            continue

        data = list(zip(images, labels, bounds))
        for img_idx in tqdm(range(len(data)), leave=False):
            all_bands, label, box = data[img_idx]

            sample_name = f"examples_{file_id}_{img_idx}"

            partition.add(partition_id[id_map[(int(file_id), img_idx)]], sample_name)
            # box = lon_top_left,lat_top_left,lon_bottom_right,lat_bottom_right
            # from_bounds = west, south, east, north
            transform = rasterio.transform.from_bounds(
                box[0], box[3], box[2], box[1], all_bands.shape[1], all_bands.shape[2]
            )

            bands = []
            for i, band in enumerate(all_bands):
                band_data = io.Band(
                    data=band,
                    band_info=io.sentinel2_13_bands[i],
                    spatial_resolution=10,
                    transform=transform,
                    crs="EPSG:4326",
                )
                bands.append(band_data)

            sample = io.Sample(bands, label=int(label), sample_name=sample_name)
            sample.save_sample(dataset_dir)

            if img_idx > 100:
                break

    partition.save(Path(dataset_dir, "original_partition.json"))


# for img_idx in tqdm(range(len(data)), leave=False):
#         image, label, box = data[img_idx]

#         img_name = f"examples_{file_id}_{img_idx}"

#         image = io.swap_band_axes_to_last(image)

#         partition.add(partition_id[id_map[(int(file_id), img_idx)]], img_name)
#         # box = lon_top_left,lat_top_left,lon_bottom_right,lat_bottom_right
#         # from_bounds = west, south, east, north
#         transform = rasterio.transform.from_bounds(box[0], box[3], box[2], box[1], image.shape[0], image.shape[1])
#         sample = io.GeoTIFF(
#             img_name,
#             image,
#             int(label),
#             10,
#             util.SENTINEL2_13_BAND_NAMES,
#             util.SENTINEL2_13_CENTRAL_WAVELENGTHS,
#             transform,
#         )

#         sample.to_geotiff(dataset_dir)
