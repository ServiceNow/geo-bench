"""
1. Install the AWS CLI. Instructions here: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
(requires sudo, so you may need to do it locally then copy the data over)
2. Make an AWS account and sign into it then navigate here: https://console.aws.amazon.com/iam/
3. Create a key pair (Access Key ID, Secret Access Key) following the instructions here: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html#cli-configure-quickstart-creds
4. Clone git repo: git clone https://github.com/kdmayer/3D-PV-Locator.git && cd 3D-PV-Locator
5. Configure AWS, follow default settings: aws configure
6. Copy imagery. Note that requester pays data transfer costs: aws s3 cp --request-payer requester s3://pv4ger/NRW_image_data/classification/ dataset/pv4ger_v1.0/
"""
import os
import sys
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from ccb import io

sys.path.append(str(Path.cwd()))


DATASET_NAME = "pv4ger"
SRC_DATASET_DIR = Path.cwd().parent.parent / io.src_datasets_dir / DATASET_NAME
DATASET_DIR = Path.cwd().parent.parent / io.datasets_dir / DATASET_NAME
SPATIAL_RESOLUTION = 0.1
PATCH_SIZE = 320
BANDS_INFO = io.make_rgb_bands(SPATIAL_RESOLUTION)


def load_sample(img_path: Path, label: int):
    # Get lat center and lon center from img path
    lat_center, lon_center = map(float, img_path.stem.split(","))
    # Lat/lons are swapped for much of the dataset, fix this.
    if lat_center < lon_center:
        lat_center, lon_center = lon_center, lat_center

    transform_center = rasterio.transform.from_origin(lon_center, lat_center, SPATIAL_RESOLUTION, SPATIAL_RESOLUTION)
    lon_corner, lat_corner = transform_center * [-PATCH_SIZE // 2, -PATCH_SIZE // 2]
    transform = rasterio.transform.from_origin(lon_corner, lat_corner, SPATIAL_RESOLUTION, SPATIAL_RESOLUTION)

    img = np.array(Image.open(img_path).convert("RGB"))

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

    return io.Sample(bands, label=label, sample_name=img_path.stem)


def convert(max_count=None, dataset_dir=DATASET_DIR):
    dataset_dir.mkdir(exist_ok=True, parents=True)

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(320, 320),
        n_time_steps=1,
        bands_info=BANDS_INFO,
        bands_stats=None,  # Will be automatically written with inspect script
        label_type=io.Classification(2, ("no solar pv", "solar pv")),
        eval_loss=io.Accuracy,
        spatial_resolution=SPATIAL_RESOLUTION,
    )
    task_specs.save(dataset_dir, overwrite=True)

    rows = []
    for split in ["train", "val", "test"]:
        for label in [0, 1]:
            split_label_dir = SRC_DATASET_DIR / split / str(label)
            for path in split_label_dir.iterdir():
                if path.suffix == ".png":
                    rows.append([split, label, path])

    df = pd.DataFrame(rows, columns=["Split", "Label", "Path"])
    df["Split"] = df["Split"].str.replace("val", "valid")
    partition = io.Partition()
    sample_count = 0
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        split = row["Split"]
        path = row["Path"]
        label = row["Label"]
        sample = load_sample(path, label)
        sample_name = path.stem
        partition.add(split, sample_name)
        sample.write(dataset_dir)
        sample_count += 1

        # temporary for creating small datasets for development purpose
        if max_count is not None and sample_count >= max_count:
            break

    partition.save(dataset_dir, "original", as_default=True)


if __name__ == "__main__":
    convert()
