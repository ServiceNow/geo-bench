import numpy as np
from rasterio import band
from torchgeo.datasets import cv4a_kenya_crop_type
from ccb.dataset import io
import datetime
from pathlib import Path
from dataset_converters import util
from tqdm import tqdm

# Notes
# * doesn't seem to have GPS coordinates
# * torchgeo doesn't seem to provide coordinates in general
# * includes a time series of 13 steps.
# * should we use the radiant mlhub api_key as a constant?


# src_dataset_dir = util.src_datasets_dir
src_dataset_dir = "/Users/alexandre.lacoste/torchgeo_datasets"  # TODO change back to default src_dataset_dir

dataset_dir = Path(util.dst_datasets_dir, "CV4AKenyaCropType")
dataset_dir.mkdir(exist_ok=True, parents=True)

band_names = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
    "CLD",
)

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


CRS = None  # TODO

BAND_INFO_LIST = [
    io.Sentinel2("01 - Coastal aerosol", ("1", "01"), 0.443),
    io.Sentinel2("02 - Blue", ("2", "02", "blue"), 0.49),
    io.Sentinel2("03 - Green", ("3", "03", "green"), 0.56),
    io.Sentinel2("04 - Red", ("4", "04", "red"), 0.665),
    io.Sentinel2("05 - Vegetation Red Edge", ("5", "05"), 0.705),
    io.Sentinel2("06 - Vegetation Red Edge", ("6", "06"), 0.74),
    io.Sentinel2("07 - Vegetation Red Edge", ("7", "07"), 0.783),
    io.Sentinel2("08 - NIR", ("8", "08", "NIR"), 0.842),
    io.Sentinel2("08A - Vegetation Red Edge", ("8A", "08A"), 0.865),
    io.Sentinel2("09 - Water vapour", ("9", "09"), 0.945),
    io.Sentinel2("11 - SWIR", ("11",), 1.61),
    io.Sentinel2("12 - SWIR", ("12",), 2.19),
    io.CloudProbability(alt_names=("CPL", "CLD")),
]


if __name__ == "__main__":

    cv4a_dataset = cv4a_kenya_crop_type.CV4AKenyaCropType(
        root="/Users/alexandre.lacoste/torchgeo_datasets",
        download=True,
        checksum=True,
        api_key="e46c4efbca1274862accc0f1616762c9c72791e00523980eea3db3c48acd106c",
        chip_size=128,
        verbose=True,
    )

    for i, tg_sample in enumerate(tqdm(cv4a_dataset)):

        field_id, x_start, y_start = cv4a_dataset.chips_metadata[i]

        images = np.array(tg_sample["image"])  # shape (n_dates, n_bands, height, width)
        mask = np.array(tg_sample["mask"])

        n_dates, n_bands, height, width = images.shape

        transform = None  # TODO

        bands = []
        for date_idx in range(n_dates):
            for band_idx in range(n_bands):
                band_data = images[date_idx, band_idx, :, :]

                band_info = BAND_INFO_LIST[band_idx]
                # if not isinstance(band_info, io.CloudProbability):
                #     band_data *= 65535

                band = io.Band(
                    data=band_data,
                    band_info=band_info,
                    date=DATES[date_idx],
                    spatial_resolution=10,
                    transform=transform,
                    crs=CRS,
                    convert_to_uint16=False,
                )
                bands.append(band)

        label = io.Band(
            data=mask, band_info=io.SegmentationMask("label"), spatial_resolution=10, transform=transform, crs=CRS
        )
        sample = io.Sample(bands, label=label, sample_name=f"field={field_id}_x={x_start:04d}_y={y_start:04d}")
        sample.save_sample(dataset_dir)

        if i > 100:
            break
