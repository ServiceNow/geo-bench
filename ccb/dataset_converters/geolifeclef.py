# pip install kaggle
# set kaggle.json according to https://www.kaggle.com/docs/api
# accept terms and conditions from: https://www.kaggle.com/c/geolifeclef-2021/data
# Download dataset using this command
# `kaggle competitions download -c geolifeclef-2021`
#


from ccb import io
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from PIL import Image
import tifffile
import rasterio

DATASET_NAME = "geolifeclef-2021"
SPATIAL_RESOLUTION = 1
PATCH_SIZE = 256
N_LABELS = 17037
SRC_DATASET_DIR = io.CCB_DIR / "source" / DATASET_NAME
DATA_PATH = Path(SRC_DATASET_DIR) / "data"
DATASET_DIR = io.CCB_DIR / "converted" / DATASET_NAME

df_suggested_landcover_alignment = pd.read_csv(DATA_PATH / "metadata" / "landcover_suggested_alignment.csv", sep=";")
landcover_labels = (
    df_suggested_landcover_alignment[["suggested_landcover_code", "suggested_landcover_label"]]
    .drop_duplicates()
    .sort_values("suggested_landcover_code")["suggested_landcover_label"]
    .values
)

# US NAIP, FR aerial based (IGN)
BAND_INFO_LIST = io.make_rgb_bands(spatial_resolution=SPATIAL_RESOLUTION)
NIR_BAND = io.SpectralBand("NIR", ("nir",), SPATIAL_RESOLUTION, wavelength=0.829)
BAND_INFO_LIST.append(NIR_BAND)
BAND_INFO_LIST.append(io.ElevationBand("Altitude", ("elevation",), spatial_resolution=SPATIAL_RESOLUTION))


def make_sample(observation_id, label, lat, lng, kaggle_sample=False):

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

    # Kaggle Sample Dataset has an extra folder called patches_sample
    kaggle_folder = "patches_sample" if kaggle_sample else ""
    filename = Path(SRC_DATASET_DIR) / "data" / kaggle_folder / region / subfolder1 / subfolder2 / observation_id

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


def convert(max_count=None, dataset_dir=DATASET_DIR, kaggle_sample=True):
    dataset_dir.mkdir(exist_ok=True, parents=True)
    partition = io.dataset.Partition()

    df_fr = pd.read_csv(DATA_PATH / "observations" / "observations_fr_train.csv", sep=";", index_col="observation_id")
    df_us = pd.read_csv(DATA_PATH / "observations" / "observations_us_train.csv", sep=";", index_col="observation_id")
    df = pd.concat((df_fr, df_us))

    # if we set a max count, we assume that the user downloaded the data from kaggle's sample patches (~3 GB)
    if kaggle_sample is True:
        # sample patches only contain the subfolder 00 (identified with the last two digits in the observation id)
        df = df.filter(regex="00$", axis=0)

    # there are no labels for the test data. Should we integrate it as well?
    # df_fr_test = pd.read_csv(DATA_PATH / "observations" / "observations_fr_test.csv", sep=";", index_col="observation_id")
    # df_us_test = pd.read_csv(DATA_PATH / "observations" / "observations_us_test.csv", sep=";", index_col="observation_id")
    # df_test = pd.concat((df_fr_test, df_us_test))
    # df = pd.concat((df, df_test))

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(PATCH_SIZE, PATCH_SIZE),
        n_time_steps=1,
        bands_info=BAND_INFO_LIST,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=io.Classification(N_LABELS),
        eval_loss=io.AccuracyTop30,
        spatial_resolution=SPATIAL_RESOLUTION,
    )

    task_specs.save(dataset_dir, overwrite=True)

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

        sample = make_sample(observation_id, int(label), latitude, longitude, kaggle_sample=kaggle_sample)
        sample.write(dataset_dir)
        partition.add(split_name, sample_name)

        # temporary for creating small datasets for development purpose
        if max_count is not None and i + 1 >= max_count:
            break

    partition.save(dataset_dir, "original", as_default=True)


if __name__ == "__main__":
    convert()