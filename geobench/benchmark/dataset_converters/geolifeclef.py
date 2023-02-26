"""GeoLifeCLEF dataset."""

# pip install kaggle
# set kaggle.json according to https://www.kaggle.com/docs/api
# accept terms and conditions from: https://www.kaggle.com/competitions/geolifeclef-2022-lifeclef-2022-fgvc9/data
# Download dataset using this command
# `kaggle competitions download -c geolifeclef-2022-lifeclef-2022-fgvc9`
#

# Note: This converter uses per default "observations_sample.csv", which can be found and copied from geolifeclef-scripts into the observations folder of geolifeclef-2022

from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import rasterio
import tifffile
from PIL import Image
from tqdm import tqdm

from geobench import io

DATASET_NAME = "geolifeclef-2022"
SPATIAL_RESOLUTION = 1
PATCH_SIZE = 256

N_LABELS = 100
SRC_DATASET_DIR = io.CCB_DIR / "source" / DATASET_NAME  # type: ignore
DATA_PATH = Path(SRC_DATASET_DIR)  # type: ignore
DATASET_DIR = io.CCB_DIR / "converted" / DATASET_NAME  # type: ignore

# US NAIP, FR aerial based (IGN)
BAND_INFO_LIST: List[Any] = io.make_rgb_bands(spatial_resolution=SPATIAL_RESOLUTION)
NIR_BAND = io.SpectralBand("NIR", ("nir",), SPATIAL_RESOLUTION, wavelength=0.829)
BAND_INFO_LIST.append(NIR_BAND)
BAND_INFO_LIST.append(io.ElevationBand("Altitude", ("elevation",), spatial_resolution=SPATIAL_RESOLUTION))


def make_sample(observation_id, label, lat, lng) -> io.Sample:
    """Create a sample.

    Args:
        observation_id:
        label:
        lat:
        lng:

    Returns:
        sample
    """
    observation_id = str(observation_id)

    region_id = observation_id[0]
    if region_id == "1":
        region = "fr"
    elif region_id == "2":
        region = "us"
    else:
        raise ValueError("Incorrect 'observation_id' {}, can not extract region id from it".format(observation_id))

    subfolder1 = observation_id[-2:]
    subfolder2 = observation_id[-4:-2]

    filename = Path(SRC_DATASET_DIR) / f"patches-{region}" / subfolder1 / subfolder2 / observation_id

    transform_center = rasterio.transform.from_origin(lng, lat, SPATIAL_RESOLUTION, SPATIAL_RESOLUTION)
    lon_corner, lat_corner = transform_center * [-PATCH_SIZE // 2, -PATCH_SIZE // 2]
    transform = rasterio.transform.from_origin(lon_corner, lat_corner, SPATIAL_RESOLUTION, SPATIAL_RESOLUTION)
    crs = "EPSG:4326"
    date = None  # ?

    bands = []

    rgb_filename = filename.with_name(filename.stem + "_rgb.jpg")
    rgb_patch = Image.open(rgb_filename)
    rgb_patch = np.asarray(rgb_patch)
    for i in range(3):
        band_data = io.Band(
            data=rgb_patch[:, :, i],
            band_info=BAND_INFO_LIST[i],
            spatial_resolution=SPATIAL_RESOLUTION,
            transform=transform,
            crs=crs,
            date=date,
            meta_info={"latitude": lat, "longitude": lng},
        )
        bands.append(band_data)

    near_ir_filename = filename.with_name(filename.stem + "_near_ir.jpg")
    near_ir_patch = Image.open(near_ir_filename)
    near_ir_patch = np.asarray(near_ir_patch)
    ir_band_data = io.Band(
        data=near_ir_patch,
        band_info=BAND_INFO_LIST[3],
        spatial_resolution=SPATIAL_RESOLUTION,
        transform=transform,
        crs=crs,
        date=date,
        meta_info={"latitude": lat, "longitude": lng},
    )
    bands.append(ir_band_data)

    altitude_filename = filename.with_name(filename.stem + "_altitude.tif")
    altitude_patch = tifffile.imread(altitude_filename)
    altitude_band_data = io.Band(
        data=altitude_patch,
        band_info=BAND_INFO_LIST[4],
        spatial_resolution=SPATIAL_RESOLUTION,
        transform=transform,
        crs=crs,
        date=date,
        meta_info={"latitude": lat, "longitude": lng},
    )
    bands.append(altitude_band_data)

    # landcover_filename = filename.with_name(filename.stem + "_landcover.tif")
    # landcover_patch = tifffile.imread(landcover_filename)
    # landcover_patch = df_suggested_landcover_alignment.values[landcover_patch]
    # landcover_band_data = io.Band(
    #         data=landcover_patch, band_info=BAND_INFO_LIST[5],
    #         spatial_resolution=SPATIAL_RESOLUTION, transform=transform, crs=crs, date=date, meta_info={'latitude': lat, 'longitude': lng})
    # bands.append(landcover_band_data)

    return io.Sample(bands, label=label, sample_name=observation_id)


def convert(max_count: int = None, dataset_dir: Path = DATASET_DIR) -> None:
    """Convert GeoLifeCLEF dataset.

    Args:
        max_count: maximum number of samples
        dataset_dir: path to dataset directory
    """
    dataset_dir.mkdir(exist_ok=True, parents=True)
    partition = io.Partition()

    observations_sample_path = Path(__file__).parent / "geolifeclef_scripts" / "observations_sample.csv"
    df = pd.read_csv(observations_sample_path, sep=";", index_col="observation_id")
    species_names_path = Path(__file__).parent / "geolifeclef_scripts" / "names.csv"
    df_species_names = pd.read_csv(species_names_path, sep=";")
    names = list(df_species_names["GBIF_species_name"])

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(PATCH_SIZE, PATCH_SIZE),
        n_time_steps=1,
        bands_info=BAND_INFO_LIST,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=io.Classification(N_LABELS, class_names=names),
        eval_loss=io.Accuracy,
        spatial_resolution=SPATIAL_RESOLUTION,
    )

    task_specs.save(str(dataset_dir), overwrite=True)

    for i, el in enumerate(tqdm(list(df.iterrows()))):
        sample_name = f"{el[0]}"

        observation_id = el[0]
        label = el[1]["species_id"]
        latitude = el[1]["latitude"]
        longitude = el[1]["longitude"]
        split_name = el[1]["subset"]
        # due to using 'valid' and not 'val
        if split_name == "val":
            split_name = "valid"

        # print(f'name={sample_name} oid={observation_id} y={label} lat={latitude} lng={longitude} split={split_name}')

        sample = make_sample(observation_id, int(label), latitude, longitude)
        sample.write(str(dataset_dir))
        partition.add(split_name, sample_name)

        # temporary for creating small datasets for development purpose
        if max_count is not None and i + 1 >= max_count:
            break

    partition.resplit_iid(split_names=("valid", "test"), ratios=(0.5, 0.5))
    partition.save(str(dataset_dir), "original", as_default=True)


if __name__ == "__main__":
    convert()
