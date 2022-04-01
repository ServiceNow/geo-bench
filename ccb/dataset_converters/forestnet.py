# Download the dataset using this link: http://download.cs.stanford.edu/deep/ForestNetDataset.zip
# (Available at this webpage: https://stanfordmlgroup.github.io/projects/forestnet/)
# Unzip the directory, then either place contents in dataset/forestnet_v1.0
# or create a symlink.
import sys
import pickle
import rasterio
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw
from pathlib import Path
from ccb import io


DATASET_NAME = "forestnet_v1.0"
SRC_DATASET_DIR = Path(io.src_datasets_dir) / DATASET_NAME
DATASET_DIR = Path(io.datasets_dir) / DATASET_NAME
SPATIAL_RESOLUTION = 15
PATCH_SIZE = 332
LABELS = [
    "Ignore",
    "Oil palm plantation",
    "Timber plantation",
    "Other large-scale plantations",
    "Grassland shrubland",
    "Small-scale agriculture",
    "Small-scale mixed plantation",
    "Small-scale oil palm plantation",
    "Mining",
    "Fish pond",
    "Logging",
    "Secondary forest",
    "Other",
]
LABEL_BAND = io.SegmentationClasses(
    "label", spatial_resolution=SPATIAL_RESOLUTION, n_classes=len(LABELS), class_names=LABELS
)
LABEL_IGNORE_VALUE = 0


def draw_img_roi(draw, shape, label: int):
    shape_type = shape.geom_type
    if shape_type == "Polygon":
        coords = np.array(shape.exterior.coords)
        draw.polygon([tuple(coord) for coord in coords], outline=label, fill=label)
    else:
        for poly in shape.geoms:
            coords = np.array(poly.exterior.coords)
            draw.polygon([tuple(coord) for coord in coords], outline=label, fill=label)


def load_sample(example_dir: Path, label: str, year: int):
    # Get lat center and lon center from img path
    lat_center, lon_center = map(float, example_dir.name.split("_"))

    transform_center = rasterio.transform.from_origin(lon_center, lat_center, SPATIAL_RESOLUTION, SPATIAL_RESOLUTION)
    lon_corner, lat_corner = transform_center * [-PATCH_SIZE // 2, -PATCH_SIZE // 2]
    transform = rasterio.transform.from_origin(lon_corner, lat_corner, SPATIAL_RESOLUTION, SPATIAL_RESOLUTION)

    # Load the visible + infrared images and add them as bands
    images_dir = example_dir / "images"
    visible_dir = images_dir / "visible"
    crs = "EPSG:4326"
    bands = []
    seen_dates = set()
    for visible_image_path in visible_dir.iterdir():
        infrared_path = Path(str(visible_image_path).replace("visible", "infrared").replace("png", "npy"))
        is_composite = visible_image_path.stem == "composite"
        if is_composite:
            # Assign last possible date to composite image
            date = f"{max(year, 2012) + 5}_01_01"
            clouds = None
        else:
            date, clouds = visible_image_path.stem.split("_cloud_")
            clouds = int(clouds)

        date = datetime.datetime.strptime(date, "%Y_%m_%d").date()
        if date in seen_dates:
            # Skip images from the same day (no way of dedup'ing them
            # in current io.dataset code)
            continue
        seen_dates.add(date)
        visible_img = np.array(Image.open(visible_image_path).convert("RGB"))
        infrared_img = np.load(infrared_path)
        meta_info = {"n_cloud_pixels": clouds, "is_composite": is_composite}
        # Visible
        for i, band_idx in enumerate([3, 2, 1]):
            band_data = io.Band(
                data=visible_img[:, :, i],
                band_info=io.landsat8_9_bands[band_idx],
                date=date,
                spatial_resolution=SPATIAL_RESOLUTION,
                transform=transform,
                crs=crs,
                meta_info=meta_info,
            )
            bands.append(band_data)

        # Infrared
        for i, band_idx in enumerate([4, 5, 6]):
            band_data = io.Band(
                data=infrared_img[:, :, i],
                band_info=io.landsat8_9_bands[band_idx],
                date=date,
                spatial_resolution=SPATIAL_RESOLUTION,  # Infrared bands have been upsampled to 15m
                transform=transform,
                crs=crs,
                meta_info=meta_info,
            )
            bands.append(band_data)

    # Create a mask with the correct unmerged label index
    forest_loss_region = example_dir / "forest_loss_region.pkl"
    with forest_loss_region.open("rb") as f:
        forest_loss_polygon = pickle.load(f)
    mask = Image.new("L", (PATCH_SIZE, PATCH_SIZE), LABEL_IGNORE_VALUE)
    label_int = LABELS.index(label)
    draw_img_roi(ImageDraw.Draw(mask), forest_loss_polygon, label_int)

    label = io.Band(
        data=np.array(mask), band_info=LABEL_BAND, spatial_resolution=SPATIAL_RESOLUTION, transform=transform, crs=crs
    )

    # How to add per-example metadata?
    # TODO: Add the year of the forest loss event
    # TODO: Load the per pixel auxiliary files and add them as bands (*.npy)
    # TODO: Load the per image auxiliary files and add them as metadata? (*.json)
    # TODO: Load all files in NCEP and add them as metadata (ncep/*)
    # aux_dir = example_dir / "auxiliary"

    return io.Sample(bands, label=label, sample_name=example_dir.name)


def convert(max_count=None, dataset_dir=DATASET_DIR):
    dataset_dir.mkdir(exist_ok=True, parents=True)

    bands_info = io.landsat8_9_bands[3:0:-1] + io.landsat8_9_bands[4:7]

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(PATCH_SIZE, PATCH_SIZE),
        n_time_steps=23,  # Variable number of time steps, max 23 across the dataset
        bands_info=bands_info,
        bands_stats=None,  # Will be automatically written with inspect script
        label_type=LABEL_BAND,
        eval_loss=io.SegmentationAccuracy,  # TODO probably not the final
        # loss eval loss. To be discussed.
        spatial_resolution=SPATIAL_RESOLUTION,
    )
    task_specs.save(dataset_dir, overwrite=True)

    partition = io.Partition()
    sample_count = 0
    for split in ["train", "val", "test"]:
        df = pd.read_csv(SRC_DATASET_DIR / f"{split}.csv")
        if split == "val":
            split = "valid"
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            example_dir = SRC_DATASET_DIR / row["example_path"]
            sample = load_sample(example_dir, row["label"], row["year"])
            sample_name = example_dir.name
            partition.add(split, sample_name)
            sample.write(dataset_dir)
            sample_count += 1

            if max_count is not None and sample_count >= max_count:
                break

        if max_count is not None and sample_count >= max_count:
            break
    partition.save(dataset_dir, "original", as_default=True)


if __name__ == "__main__":
    convert()
