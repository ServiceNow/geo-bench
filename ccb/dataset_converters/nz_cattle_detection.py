# Downloaded from "https://zenodo.org/record/5908869"

# to authors
# * coordintates are lon-lat (not lat-lon)
# * specify the coordintates are for the center.
# * can we change "Kapiti_Coast" to "Kapiti-Coast"

from ccb import io
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import rasterio
import datetime

DATASET_NAME = "nz_cattle"
SRC_DATASET_DIR = Path(io.src_datasets_dir, DATASET_NAME)
DATASET_DIR = Path(io.datasets_dir, DATASET_NAME)


BAND_INFO_LIST = io.make_rgb_bands(0.1)


def parse_file_name(name):
    name = name.replace("Kapiti_Coast", "Kapiti-Coast")
    _index, location, year, lon_lat = name.split("_")[:4]
    lon_center, lat_center = [float(val) for val in lon_lat.split(",")]
    year = int(year[1:-1].split("-")[-1])
    date = datetime.date(year=year, month=1, day=1)
    transform_center = rasterio.transform.from_origin(lon_center, lat_center, 0.1, 0.1)
    lon_corner, lat_corner = transform_center * [-250, -250]
    transform = rasterio.transform.from_origin(lon_corner, lat_corner, 0.1, 0.1)

    crs = rasterio.crs.CRS.from_epsg(4326)

    return location, date, transform, crs


def load_sample(img_path: Path):
    label_path = img_path.with_suffix(".png.mask.0.txt")
    with Image.open(img_path) as im:
        data = np.array(im)[:, :, :3]

    location, date, transform, crs = parse_file_name(img_path.stem)
    coords = []
    with open(label_path, "r") as fd:
        for line in fd:
            coord = [int(val) for val in line.split(",")]
            coords.append(coord)

    bands = []
    for i in range(3):
        band_data = io.Band(
            data=data[:, :, i],
            band_info=BAND_INFO_LIST[i],
            spatial_resolution=0.1,
            transform=transform,
            crs=crs,
            date=date,
            meta_info={"location": location},
        )
        bands.append(band_data)

    return io.Sample(bands, label=coords, sample_name=img_path.stem)


def convert(max_count=None, dataset_dir=DATASET_DIR):
    dataset_dir.mkdir(exist_ok=True, parents=True)

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(500, 500),
        n_time_steps=1,
        bands_info=BAND_INFO_LIST,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=io.PointAnnotation(),
        eval_loss=io.SegmentationAccuracy(),  # TODO decide on the loss
        spatial_resolution=0.1,
    )
    task_specs.save(dataset_dir, overwrite=True)
    partition = io.Partition()

    path_list = list(Path(SRC_DATASET_DIR, "cow_images").iterdir())

    sample_count = 0
    partition = io.Partition()  # default partition: everything in train
    for file in tqdm(path_list):
        if file.suffix == ".png":
            sample = load_sample(img_path=file)
            sample.write(dataset_dir)

            partition.add("train", sample.sample_name)

            sample_count += 1
            if max_count is not None and sample_count >= max_count:
                break
    partition.save(dataset_dir, "nopartition", as_default=True)


if __name__ == "__main__":
    convert()
