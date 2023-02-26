"""SeasonNet dataset."""
import datetime
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from geobench import io

# change dimensions to be H, W, C
# Paths
DATASET_NAME = "seasonet"
SRC_DATASET_DIR = Path(io.src_datasets_dir, DATASET_NAME)  # type: ignore
DATASET_DIR = Path(io.datasets_dir, DATASET_NAME)  # type: ignore

LABELS = [
    "background",
    "Continuous urban fabric",
    "Discontinuous urban fabric",
    "Industrial or commercial units",
    "Road and rail networks and associated land",
    "Port areas",
    "Airports",
    "Mineral extraction sites",
    "Dump sites",
    "Construction sites",
    "Green urban areas",
    "Sport and leisure facilities",
    "Non-irrigated arable land",
    "Vineyards",
    "Fruit trees and berry plantations",
    "Pastures",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grasslands",
    "Moors and heathland",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Bare rock",
    "Sparsely vegetated areas",
    "Inland marshes",
    "Peat bogs",
    "Salt marshes",
    "Intertidal flats",
    "Water courses",
    "Water bodies",
    "Coastal lagoons",
    "Estuaries",
    "Sea and ocean",
]

BAND_INFO_LIST: List[Any] = io.sentinel2_13_bands[:]
dropped_band = BAND_INFO_LIST.pop(10)
assert dropped_band.name == "10 - SWIR - Cirrus"

SPATIAL_RESOLUTION = 10

LABEL_BAND = io.SegmentationClasses(
    "label", spatial_resolution=SPATIAL_RESOLUTION, n_classes=len(LABELS), class_names=LABELS
)

DATASET_DIR = Path(io.CCB_DIR, "converted", DATASET_NAME)

HEIGHT = 120
WIDTH = 120

# there are multiple seasons and snow layer available
# specify from which you want to sample
SEASONS = ["Fall"]

train_split = "/mnt/data/cc_benchmark/source/seasonet/splits/train.csv"
val_split = "/mnt/data/cc_benchmark/source/seasonet/splits/val.csv"
test_split = "/mnt/data/cc_benchmark/source/seasonet/splits/test.csv"


split_paths = list((SRC_DATASET_DIR / "splits").glob("*.csv"))
SPLIT_DICT = {}
for path in split_paths:
    split = str(path).split("/")[-1].split(".")[0]
    df = pd.read_csv(path, header=None)
    for id in df.loc[:, 0].tolist():
        SPLIT_DICT[id] = split


def load_bands(path, band_info_list):
    """Load bands from tif files.

    Args:
        path: path to tif file
        band_info_list: corresponding band_info to order of bands

    Returns:
        dictionary mapping band info to band
    """
    with rasterio.open(path) as src:
        data = src.read()

        band_dict = {}
        for i, band_info in enumerate(band_info_list):
            band = io.Band(
                data=data[i, :, :],
                band_info=band_info,
                date=None,
                spatial_resolution=src.res[0],
                transform=src.transform,
                crs=src.crs,
                convert_to_int16=False,
            )
            band_dict[band_info] = band

    return band_dict


def load_label_as_band(label_path):
    """Load the label as band.

    Args:
        label_path: path to label tif file

    Returns:
        io.Band of label
    """
    with rasterio.open(label_path) as src:
        label = io.Band(
            data=src.read().transpose((1, 2, 0)),
            band_info=LABEL_BAND,
            spatial_resolution=src.res[0],
            transform=src.transform,
            crs=src.crs,
        )

    return label


def convert(max_count=None, dataset_dir=DATASET_DIR) -> None:
    """Convert SeasoNet dataset.

    Args:
        max_count: maximum number of samples
        dataset_dir: path to dataset directory
    """
    dataset_dir.mkdir(exist_ok=True, parents=True)

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(HEIGHT, WIDTH),
        n_time_steps=1,
        bands_info=BAND_INFO_LIST,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=LABEL_BAND,
        eval_loss=io.SegmentationAccuracy,  # TODO probably not the final loss eval loss. To be discussed.
        # either 50cm or 40cm, Airbus Pleiades 50cm, https://radiantearth.blob.core.windows.net/mlhub/technoserve-cashew-benin/Documentation.pdf
        spatial_resolution=SPATIAL_RESOLUTION,
    )

    partition = io.Partition()

    task_specs.save(dataset_dir, overwrite=True)

    # load the metafile from which to load samples
    meta_df = pd.read_csv(SRC_DATASET_DIR / "meta.csv")
    meta_df = meta_df[meta_df["Season"].isin(SEASONS)]

    # only use one grid
    meta_df = meta_df[meta_df["Grid"] == 1]
    print(len(meta_df))

    # only consider cloud free and non snow-images
    meta_df = meta_df[(meta_df["Clouds"] == 0.0) & (meta_df["Snow"] == 0.0)]

    # sample max_count number of samples from df
    meta_df = meta_df.sample(n=max_count, random_state=1).reset_index(drop=True)

    # iterate over df to load samples
    for idx, row in tqdm(meta_df.iterrows()):

        sample_dir = SRC_DATASET_DIR / row.Path

        id = str(sample_dir).split("/")[-1]

        band_dict = {}
        rgb_band_info = [BAND_INFO_LIST[3], BAND_INFO_LIST[2], BAND_INFO_LIST[1]]
        band_dict.update(load_bands(sample_dir / (id + "_10m_RGB.tif"), rgb_band_info))
        band_dict.update(load_bands(sample_dir / (id + "_10m_IR.tif"), [BAND_INFO_LIST[7]]))

        vegetation_swir_info = [
            BAND_INFO_LIST[4],
            BAND_INFO_LIST[5],
            BAND_INFO_LIST[6],
            BAND_INFO_LIST[8],
            BAND_INFO_LIST[10],
            BAND_INFO_LIST[11],
        ]
        band_dict.update(load_bands(sample_dir / (id + "_20m.tif"), vegetation_swir_info))

        water_info = [
            BAND_INFO_LIST[0],
            BAND_INFO_LIST[9],
        ]
        band_dict.update(load_bands(sample_dir / (id + "_60m.tif"), water_info))

        ordered_bands = [band_dict[band_info] for band_info in BAND_INFO_LIST]

        label = load_label_as_band(sample_dir / (id + "_labels.tif"))

        sample = io.Sample(ordered_bands, label=label, sample_name=SPLIT_DICT[int(id.split("_")[-1])])
        sample.write(dataset_dir)
        partition.add(split, id)

    partition.save(dataset_dir, "default")


if __name__ == "__main__":

    convert(max_count=100)
