"""SeasonNet dataset."""
import datetime
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from ccb import io

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


def split_rgb_bands(rgb_path: Path):
    """Split the concatenated rgb bands.

    Args:
        rgb_path: path to tif file of concatentated rgb image

    Returns:
        rgb io.Bands
    """
    with rasterio.open(rgb_path) as src:
        data = src.read()

        r_data, g_data, b_data = data[0, :, :], data[1, :, :], data[2, :, :]

        red_band = io.Band(
            data=r_data[..., None],
            band_info=BAND_INFO_LIST[3],
            date=None,
            spatial_resolution=src.res[0],
            transform=src.transform,
            crs=src.crs,
            convert_to_int16=False,
        )

        green_band = io.Band(
            data=g_data[..., None],
            band_info=BAND_INFO_LIST[2],
            date=None,
            spatial_resolution=src.res[0],
            transform=src.transform,
            crs=src.crs,
            convert_to_int16=False,
        )

        blue_band = io.Band(
            data=b_data[..., None],
            band_info=BAND_INFO_LIST[1],
            date=None,
            spatial_resolution=src.res[0],
            transform=src.transform,
            crs=src.crs,
            convert_to_int16=False,
        )

    return red_band, green_band, blue_band


def create_nir_band(nir_path: Path):
    """Convert nir to io.Band.

    Args:
        nir_path: path to nir tif file

    Returns:
        io.Band of nir band
    """
    with rasterio.open(nir_path) as src:
        data = src.read()

        nir_band = io.Band(
            data=data[..., None],
            band_info=BAND_INFO_LIST[7],
            date=None,
            spatial_resolution=src.res[0],
            transform=src.transform,
            crs=src.crs,
            convert_to_int16=False,
        )

    return nir_band


def split_vegetation_swir_bands(veg_and_swir_path):
    """Split the concatenated vegetation and swir bands.

    Args:
        veg_and_swir_path: path to tif with vegetation and swir bands

    Returns:
        io.Bands
    """
    with rasterio.open(veg_and_swir_path) as src:
        data = src.read()

        band_5 = io.Band(
            data=data[0, :, :][..., None],
            band_info=BAND_INFO_LIST[4],
            date=None,
            spatial_resolution=src.res[0],
            transform=src.transform,
            crs=src.crs,
            convert_to_int16=False,
        )

        band_6 = io.Band(
            data=data[1, :, :][..., None],
            band_info=BAND_INFO_LIST[5],
            date=None,
            spatial_resolution=src.res[0],
            transform=src.transform,
            crs=src.crs,
            convert_to_int16=False,
        )

        band_7 = io.Band(
            data=data[2, :, :][..., None],
            band_info=BAND_INFO_LIST[6],
            date=None,
            spatial_resolution=src.res[0],
            transform=src.transform,
            crs=src.crs,
            convert_to_int16=False,
        )

        band_8a = io.Band(
            data=data[3, :, :][..., None],
            band_info=BAND_INFO_LIST[8],
            date=None,
            spatial_resolution=src.res[0],
            transform=src.transform,
            crs=src.crs,
            convert_to_int16=False,
        )

        band_11 = io.Band(
            data=data[4, :, :][..., None],
            band_info=BAND_INFO_LIST[10],
            date=None,
            spatial_resolution=src.res[0],
            transform=src.transform,
            crs=src.crs,
            convert_to_int16=False,
        )

        band_12 = io.Band(
            data=data[5, :, :][..., None],
            band_info=BAND_INFO_LIST[11],
            date=None,
            spatial_resolution=src.res[0],
            transform=src.transform,
            crs=src.crs,
            convert_to_int16=False,
        )

    return band_5, band_6, band_7, band_8a, band_11, band_12


def split_water_bands(water_band_path):
    """Split the concatenated water bands.

    Args:
        water_band_path: path to tif file with water bands

    Returns:
        io.Bands
    """
    with rasterio.open(water_band_path) as src:
        data = src.read()

        band_1 = io.Band(
            data=data[0, :, :][..., None],
            band_info=BAND_INFO_LIST[0],
            date=None,
            spatial_resolution=src.res[0],
            transform=src.transform,
            crs=src.crs,
            convert_to_int16=False,
        )

        band_9 = io.Band(
            data=data[1, ...][..., None],
            band_info=BAND_INFO_LIST[9],
            date=None,
            spatial_resolution=src.res[0],
            transform=src.transform,
            crs=src.crs,
            convert_to_int16=False,
        )

    return band_1, band_9


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

        split = np.random.choice(("train", "valid", "test"), p=(0.8, 0.1, 0.1))

        sample_dir = SRC_DATASET_DIR / row.Path

        id = str(sample_dir).split("/")[-1]
        rgb_path = sample_dir / (id + "_10m_RGB.tif")
        red_band_4, green_band_3, blue_band_2 = split_rgb_bands(rgb_path)

        nir_path = sample_dir / (id + "_10m_IR.tif")
        nir_band_8 = create_nir_band(nir_path)

        veg_and_swir_path = sample_dir / (id + "_20m.tif")
        band_5, band_6, band_7, band_8a, band_11, band_12 = split_vegetation_swir_bands(veg_and_swir_path)

        water_band_path = sample_dir / (id + "_60m.tif")
        band_1, band_9 = split_water_bands(water_band_path)

        label_path = sample_dir / (id + "_labels.tif")
        label = load_label_as_band(label_path)

        bands = [
            band_1,
            blue_band_2,
            green_band_3,
            red_band_4,
            band_5,
            band_6,
            band_7,
            nir_band_8,
            band_8a,
            band_9,
            band_11,
            band_12,
        ]

        if label.data.min() == 0:
            import pdb

            pdb.set_trace()

        sample = io.Sample(bands, label=label, sample_name=id)
        sample.write(dataset_dir)
        partition.add(split, id)

    partition.save(dataset_dir, "default")


if __name__ == "__main__":
    convert(max_count=100)
