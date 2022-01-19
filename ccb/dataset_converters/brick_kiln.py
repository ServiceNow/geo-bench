# Downloaded from "https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg13/brick_kiln.html"
# Try this command for downloading on headless server:
#   wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1aOWHRY72LlHNv7nwbAcPEHWcuuYnaEtx' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1aOWHRY72LlHNv7nwbAcPEHWcuuYnaEtx" -O brick_kiln_v1.0.tar.gz && rm -rf /tmp/cookies.txt


from ccb.io import dataset, task
import numpy as np
import csv
from ccb.dataset_converters import util
import rasterio
from pathlib import Path
import h5py
from tqdm import tqdm

DATASET_NAME = "brick_kiln_v1.0"
SRC_DATASET_DIR = Path(dataset.src_datasets_dir, DATASET_NAME)
DATASET_DIR = Path(dataset.datasets_dir, DATASET_NAME)


def load_examples_bloc(file_path):
    """Load a .h5py bloc of images with their labels."""
    file_id = file_path.stem.split("_")[1]

    with h5py.File(file_path) as data:

        images = data["images"][:]
        labels = data["labels"][:]
        bounds = data["bounds"][:]

        return images, labels, bounds, file_id


def read_list_eval_partition(csv_file):
    "The CSV file contains redundant information and the information for the original partition."
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


def make_sample(src_bands, label, coord_box, sample_name):
    """Converts the data in src_bands. Instantiate each Band separately and combine them into Sample
    """

    lon_top_left, lat_top_left, lon_bottom_right, lat_bottom_right = coord_box
    transform = rasterio.transform.from_bounds(
        west=lon_top_left,
        south=lat_bottom_right,
        east=lon_bottom_right,
        north=lat_top_left,
        width=src_bands.shape[1],
        height=src_bands.shape[2],
    )

    bands = []
    for i, band in enumerate(src_bands):
        band_data = dataset.Band(
            data=band, band_info=dataset.sentinel2_13_bands[i],
            spatial_resolution=10, transform=transform, crs="EPSG:4326",)
        bands.append(band_data)

    return dataset.Sample(bands, label=int(label), sample_name=sample_name)


def convert(max_count=None, dataset_dir=DATASET_DIR):
    dataset_dir.mkdir(exist_ok=True, parents=True)

    task_specs = task.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(64, 64),
        n_time_steps=1,
        bands_info=dataset.sentinel2_13_bands,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=task.Classification(2, ("not brick kiln", "brick kiln")),
        eval_loss=task.Accuracy,
        spatial_resolution=10,
    )
    task_specs.save(dataset_dir)

    _, partition_id, _, _, _, _, id_map = read_list_eval_partition(Path(SRC_DATASET_DIR, "list_eval_partition.csv"))

    file_list = list(SRC_DATASET_DIR.iterdir())
    partition = dataset.Partition(map={0: "train", 1: "valid", 2: "test"})

    sample_count = 0
    for file_idx, file_path in enumerate(tqdm(file_list)):
        if file_path.suffix != ".hdf5":
            continue

        # In this dataset, images ares stored as a batch of up to 999 samples, but sometime there are none.
        images, labels, bounds, file_id = load_examples_bloc(file_path)

        if images.shape[0] == 0:
            print("Skipping block of shape 0. Shape = %s" % (str(images.shape)))
            continue

        data = list(zip(images, labels, bounds))
        for img_idx in tqdm(range(len(data)), leave=False):
            all_bands, label, coord_box = data[img_idx]
            sample_name = f"examples_{file_id}_{img_idx}"

            partition.add(partition_id[id_map[(int(file_id), img_idx)]], sample_name)
            sample = make_sample(all_bands, label, coord_box, sample_name)
            sample.write(dataset_dir)
            sample_count += 1
            # temporary for creating small datasets for development purpose
            if max_count is not None and sample_count >= max_count:
                break

        if max_count is not None and sample_count >= max_count:
            break
    partition.save(dataset_dir, "original_partition")


if __name__ == "__main__":
    convert()
