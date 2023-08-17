"""PV4GER dataset.

1. Install the AWS CLI. Instructions here: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
(requires sudo, so you may need to do it locally then copy the data over)
2. Make an AWS account and sign into it then navigate here: https://console.aws.amazon.com/iam/
3. Create a key pair (Access Key ID, Secret Access Key) following the instructions here: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html#cli-configure-quickstart-creds
4. Clone git repo: git clone https://github.com/kdmayer/3D-PV-Locator.git && cd 3D-PV-Locator
5. Configure AWS, follow default settings: aws configure
6. Copy imagery. Note that requester pays data transfer costs: aws s3 cp --request-payer requester s3://pv4ger/NRW_image_data/{classification,segmentation}/ dataset/pv4ger_v1.0/
"""
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from tqdm import tqdm

from geobench import io
from geobench.benchmark.dataset_converters.util import center_to_transform
from geobench.io.dataset import Band, Sample

sys.path.append(str(Path.cwd()))


DATASET_NAME = "pv4ger"
SRC_DATASET_DIR = io.GEO_BENCH_DIR / "source" / DATASET_NAME  # type: ignore
# CLS_DATASET_DIR = io.CCB_DIR / "converted" / f"{DATASET_NAME}_classification"
# SEG_DATASET_DIR = io.CCB_DIR / "converted" / f"{DATASET_NAME}_segmentation"
DATASET_DIR = io.GEO_BENCH_DIR / "converted" / f"{DATASET_NAME}_classification"  # type: ignore
SPATIAL_RESOLUTION = 0.1
PATCH_SIZE = 320
BANDS_INFO = io.make_rgb_bands(SPATIAL_RESOLUTION)
LABELS = ["no solar pv", "solar pv"]
SEG_LABEL_BAND = io.SegmentationClasses("label", spatial_resolution=SPATIAL_RESOLUTION, n_classes=2, class_names=LABELS)


def get_transform(img_path: str):
    """Create transform based on image.

    Args:
        img_path: path to image

    Returns:
        rasterio transform
    """
    # Get lat center and lon center from img path
    lat_center, lon_center = map(float, Path(img_path).stem.split(","))
    # Lat/lons are swapped for much of the dataset, fix this.
    if lat_center < lon_center:
        lat_center, lon_center = lon_center, lat_center

    radius_in_meter = PATCH_SIZE / 2 * SPATIAL_RESOLUTION
    transform = center_to_transform(lat_center, lon_center, radius_in_meter, (PATCH_SIZE, PATCH_SIZE))

    # transform_center = rasterio.transform.from_origin(lon_center, lat_center, SPATIAL_RESOLUTION, SPATIAL_RESOLUTION)
    # lon_corner, lat_corner = transform_center * [-PATCH_SIZE // 2, -PATCH_SIZE // 2]
    # transform = rasterio.transform.from_origin(lon_corner, lat_corner, SPATIAL_RESOLUTION, SPATIAL_RESOLUTION)

    return transform


def get_bands(img: "np.typing.NDArray[np.int_]", transform) -> List[Band]:
    """Retrieve RGB bands.

    Args:
        img: image array
        transform: transformation applied to image

    Returns:
        list of retrieved bands
    """
    bands = []
    for i in range(3):
        band_data = io.Band(
            data=img[:, :, i],
            band_info=BANDS_INFO[i],
            spatial_resolution=SPATIAL_RESOLUTION,
            transform=transform,
            crs="EPSG:4326",
        )
        bands.append(band_data)

    return bands


def load_cls_sample(img_path: Path, label: int) -> Sample:
    """Create classification sample.

    Args:
        img_path: path to image
        label: classification label

    Returns:
        classification sample
    """
    transform = get_transform(str(img_path))

    img = np.array(Image.open(img_path).convert("RGB"))

    bands = get_bands(img, transform)

    return io.Sample(bands, label=label, sample_name=img_path.stem)


def load_seg_sample(img_path: Path, mask_path: Path) -> Sample:
    """Create segmentation sample.

    Args:
        img_path: path to image
        mask_path: path to image mask

    Returns:
        segmentation sample
    """
    transform = get_transform(str(img_path))

    img = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))
    mask[mask > 0] = 1

    bands = get_bands(img, transform)

    label = io.Band(
        data=mask, band_info=SEG_LABEL_BAND, spatial_resolution=SPATIAL_RESOLUTION, transform=transform, crs="EPSG:4326"
    )

    return io.Sample(bands, label=label, sample_name=img_path.stem)


def convert(max_count: int = None, dataset_dir: Path = DATASET_DIR, classification: bool = True) -> None:
    """Convert pv4ger dataset.

    Args:
        max_count: maximum number of samples
        dataset_dir: path to dataset directory
        classification: whether or not to convert the classification version
    """
    if classification:
        label_type = io.Classification(2, LABELS)  # type: ignore
        # dataset_dir = CLS_DATASET_DIR
    else:
        label_type = SEG_LABEL_BAND  # type: ignore
        dataset_dir = dataset_dir.with_name(f"{DATASET_NAME}_segmentation")

    dataset_dir.mkdir(exist_ok=True, parents=True)

    task_specs = io.TaskSpecifications(
        dataset_name=dataset_dir.name,
        patch_size=(PATCH_SIZE, PATCH_SIZE),
        n_time_steps=1,
        bands_info=BANDS_INFO,
        bands_stats=None,  # Will be automatically written with inspect script
        label_type=label_type,
        spatial_resolution=SPATIAL_RESOLUTION,
    )
    task_specs.save(str(dataset_dir), overwrite=True)

    rows = []

    if classification:
        for split in ["train", "val", "test"]:
            for label in [0, 1]:
                split_label_dir = SRC_DATASET_DIR / split / str(label)
                for path in split_label_dir.iterdir():
                    if path.suffix == ".png":
                        rows.append([split, label, path])
    else:
        for split in ["train", "val", "test"]:
            split_dir = SRC_DATASET_DIR / split / "image"
            for image_path in split_dir.iterdir():
                if image_path.suffix == ".png":
                    mask_path = image_path.parent.parent / "mask" / image_path.name
                    rows.append([split, mask_path, image_path])

    df = pd.DataFrame(rows, columns=["Split", "Label", "Path"])
    df["Split"] = df["Split"].str.replace("val", "valid")

    partition = io.Partition()
    sample_count = 0
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        split = row["Split"]
        row_path: Path = row["Path"]
        if classification:
            sample = load_cls_sample(row_path, row["Label"])
        else:
            sample = load_seg_sample(row_path, row["Label"])
        sample_name = row_path.stem
        partition.add(split, sample_name)
        sample.write(str(dataset_dir))
        sample_count += 1

        # temporary for creating small datasets for development purpose
        if max_count is not None and sample_count >= max_count:
            break

    partition.save(str(dataset_dir), "original", as_default=True)


if __name__ == "__main__":
    convert(classification=True)
    # convert(classification=False)
