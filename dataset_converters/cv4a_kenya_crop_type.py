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

# Notes
# * doesn't seem to have GPS coordinates
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


BAND_INFO_LIST = io.sentinel2_13_bands[:]
dropped_band = BAND_INFO_LIST.pop(10)
assert dropped_band.name == "10 - SWIR - Cirrus"
BAND_INFO_LIST.append(io.CloudProbability(alt_names=("CPL", "CLD")),)

LABEL_BAND = io.SegmentationClasses("label", spatial_resolution=10, n_classes=7)


def make_sample(images, mask, sample_name):
    n_dates, n_bands, _height, _width = images.shape

    transform = None  # TODO can't find the GPS coordinates from torch geo.
    crs = None

    bands = []
    for date_idx in range(n_dates):
        for band_idx in range(n_bands):
            band_data = images[date_idx, band_idx, :, :]

            band_info = BAND_INFO_LIST[band_idx]

            band = io.Band(
                data=band_data,
                band_info=band_info,
                date=DATES[date_idx],
                spatial_resolution=10,
                transform=transform,
                crs=crs,
                convert_to_int16=False,
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

    for i, tg_sample in enumerate(tqdm(cv4a_dataset)):

        field_id, x_start, y_start = cv4a_dataset.chips_metadata[i]
        sample_name = f"field={field_id}_x={x_start:04d}_y={y_start:04d}"

        images = np.array(tg_sample["image"])
        mask = np.array(tg_sample["mask"])

        sample = make_sample(images, mask, sample_name)
        sample.save_sample(dataset_dir)

        # temporary for creating small datasets for development purpose
        if i > 100:
            break
