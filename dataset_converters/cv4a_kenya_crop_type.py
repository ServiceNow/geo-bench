import numpy as np
from rasterio import band
from torchgeo.datasets import cv4a_kenya_crop_type
from ccb.dataset import io
import datetime
from pathlib import Path
from dataset_converters import util
from tqdm import tqdm
from toolbox.core.task_specs import TaskSpecifications
from toolbox.core.loss import SegmentationAccuracy


# Deprecated:
# we need to re-write this scripts so that it can properly splits into train / test
# and extract georefence. torchgeo is not an option.

# Notes
# * torchgeo doesn't seem to provide coordinates in general as a general interface
# * should we use the radiant mlhub api_key as a constant?


src_dataset_dir = util.src_datasets_dir

dataset_dir = Path(util.dst_datasets_dir, "CV4AKenyaCropType")
dataset_dir.mkdir(exist_ok=True, parents=True)

DATES = [
    "20190606",
    "20190701",
    "20190706",
    "20190711",
    "20190721",
    "20190805",
    "20190815",
    "20190825",
    "20190909",
    "20190919",
    "20190924",
    "20191004",
    "20191103",
]


DATES = [datetime.datetime.strptime(date, "%Y%m%d").date() for date in DATES]

max_band_value = {
    "06 - Vegetation Red Edge": 1.4976,
    "02 - Blue": 1.7024,
    "03 - Green": 1.6,
    "12 - SWIR": 1.2458,
    "05 - Vegetation Red Edge": 1.5987,
    "04 - Red": 1.5144,
    "01 - Coastal aerosol": 1.7096,
    "07 - Vegetation Red Edge": 1.4803,
    "11 - SWIR": 1.0489,
    "09 - Water vapour": 1.6481,
    "08A - Vegetation Red Edge": 1.4244,
    "08 - NIR": 1.4592,
}

BAND_INFO_LIST = io.sentinel2_13_bands[:]
dropped_band = BAND_INFO_LIST.pop(10)
assert dropped_band.name == "10 - SWIR - Cirrus"
BAND_INFO_LIST.append(io.CloudProbability(alt_names=("CPL", "CLD")),)

LABEL_BAND = io.SegmentationClasses("label", spatial_resolution=10, n_classes=8)


def make_sample(images, mask, sample_name):
    n_dates, n_bands, _height, _width = images.shape

    transform = None  # TODO can't find the GPS coordinates from torch geo.
    crs = None

    bands = []
    for date_idx in range(n_dates):
        for band_idx in range(n_bands):
            band_data = images[date_idx, band_idx, :, :]

            band_info = BAND_INFO_LIST[band_idx]

            if band_info.name in max_band_value:
                band_data = band_data / max_band_value[band_info.name] * 10000

            band = io.Band(
                data=band_data,
                band_info=band_info,
                date=DATES[date_idx],
                spatial_resolution=10,
                transform=transform,
                crs=crs,
                # convert_to_int16=False,
            )
            bands.append(band)

    label = io.Band(data=mask, band_info=LABEL_BAND, spatial_resolution=10, transform=transform, crs=crs)
    return io.Sample(bands, label=label, sample_name=sample_name)


if __name__ == "__main__":

    cv4a_dataset = cv4a_kenya_crop_type.CV4AKenyaCropType(
        root=src_dataset_dir,
        download=True,
        checksum=True,
        api_key="e46c4efbca1274862accc0f1616762c9c72791e00523980eea3db3c48acd106c",
        chip_size=128,
        verbose=True,
    )

    task_specs = TaskSpecifications(
        dataset_name="CV4AKenyaCropType",
        patch_size=(128, 128),
        n_time_steps=13,
        bands_info=BAND_INFO_LIST,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=LABEL_BAND,
        eval_loss=SegmentationAccuracy,  # TODO probably not the final loss eval loss. To be discussed.
        spatial_resolution=10,
    )
    task_specs.save(dataset_dir)

    set_map = {}
    trn, tst = cv4a_dataset.get_splits()

    for id in trn:
        set_map[id] = 0
    for id in tst:
        set_map[id] = 1
    set_map[0] = 0

    j = 0
    for i, tg_sample in enumerate(tqdm(cv4a_dataset)):

        if np.all(np.array(tg_sample["field_ids"]) == 0):
            continue

        tile_id, x_start, y_start = cv4a_dataset.chips_metadata[i]
        sample_name = f"tile={tile_id}_x={x_start:04d}_y={y_start:04d}"
        uids = np.unique(tg_sample["field_ids"])

        images = np.array(tg_sample["image"])
        mask = np.array(tg_sample["mask"])

        set_count = np.bincount([set_map[id] for id in uids])
        print(set_count)

        sample = make_sample(images, mask, sample_name)
        sample.save_sample(dataset_dir)

        j += 1
        # temporary for creating small datasets for development purpose
        if j >= 100:
            break
