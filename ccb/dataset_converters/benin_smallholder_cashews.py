# Smallholder Cashew Dataset will be downloaded by torchgeo
#
# 1) This requires Radiant MLHub package and API token
#   pip install radiant_mlhub
# 2) Sign up for a MLHub account here: https://mlhub.earth/
# 3) Type this in your terminal:
#   mlhub configure
# and enter your API key.
#
# More info on the dataset: https://mlhub.earth/10.34911/rdnt.hfv20i

from ccb import io
from pathlib import Path
from tqdm import tqdm
from torchgeo.datasets import BeninSmallHolderCashews
import datetime
import os
from multiprocessing import Pool

# Classification labels
LABELS = (
    "no data",
    "well-managed plantation",
    "poorly-managed plantation",
    "non-plantation",
    "residential",
    "background",
    "uncertain",
)
DATES = (
    "2019-11-05",
    "2019-11-10",
    "2019-11-15",
    "2019-11-20",
    "2019-11-30",
    "2019-12-05",
    "2019-12-10",
    "2019-12-15",
    "2019-12-20",
    "2019-12-25",
    "2019-12-30",
    "2020-01-04",
    "2020-01-09",
    "2020-01-14",
    "2020-01-19",
    "2020-01-24",
    "2020-01-29",
    "2020-02-08",
    "2020-02-13",
    "2020-02-18",
    "2020-02-23",
    "2020-02-28",
    "2020-03-04",
    "2020-03-09",
    "2020-03-14",
    "2020-03-19",
    "2020-03-24",
    "2020-03-29",
    "2020-04-03",
    "2020-04-08",
    "2020-04-13",
    "2020-04-18",
    "2020-04-23",
    "2020-04-28",
    "2020-05-03",
    "2020-05-08",
    "2020-05-13",
    "2020-05-18",
    "2020-05-23",
    "2020-05-28",
    "2020-06-02",
    "2020-06-07",
    "2020-06-12",
    "2020-06-17",
    "2020-06-22",
    "2020-06-27",
    "2020-07-02",
    "2020-07-07",
    "2020-07-12",
    "2020-07-17",
    "2020-07-22",
    "2020-07-27",
    "2020-08-01",
    "2020-08-06",
    "2020-08-11",
    "2020-08-16",
    "2020-08-21",
    "2020-08-26",
    "2020-08-31",
    "2020-09-05",
    "2020-09-10",
    "2020-09-15",
    "2020-09-20",
    "2020-09-25",
    "2020-09-30",
    "2020-10-10",
    "2020-10-15",
    "2020-10-20",
    "2020-10-25",
    "2020-10-30",
)
DATES = [datetime.datetime.strptime(date, "%Y-%m-%d").date() for date in DATES]


SPATIAL_RESOLUTION = 0.5  # meters, to be confirmed
N_TIMESTEPS = 70
LABEL_BAND = io.SegmentationClasses("label", spatial_resolution=SPATIAL_RESOLUTION, n_classes=len(LABELS))

# Paths
DATASET_NAME = "smallholder_cashew"
SRC_DATASET_DIR = Path(io.src_datasets_dir, DATASET_NAME)
DATASET_DIR = Path(io.datasets_dir, DATASET_NAME)


def make_sample(images, mask, sample_name):
    n_timesteps, n_bands, _height, _width = images.shape

    transform = None  # TODO can't find the GPS coordinates from torch geo.
    crs = None

    bands = []
    for date_idx in range(n_timesteps):

        for band_idx in range(n_bands):
            band_data = images[date_idx, band_idx, :, :]

            band_info = io.sentinel2_13_bands[band_idx]

            band = io.Band(
                data=band_data,
                band_info=band_info,
                date=DATES[date_idx],
                spatial_resolution=SPATIAL_RESOLUTION,
                transform=transform,
                crs=crs,
                convert_to_int16=False,
            )
            bands.append(band)

    label = io.Band(
        data=mask, band_info=LABEL_BAND, spatial_resolution=SPATIAL_RESOLUTION, transform=transform, crs=crs
    )
    return io.Sample(bands, label=label, sample_name=sample_name)


def process_i(args):
    i, tg_sample, dataset_dir = args
    sample_name = f"sample_{i:08d}"

    images = tg_sample["image"].numpy()
    mask = tg_sample["mask"].numpy()

    sample = make_sample(images, mask, sample_name)
    sample.write(dataset_dir)
    # print(f'Wrote {sample_name}')
    return i


def convert(max_count=None, dataset_dir=DATASET_DIR):
    dataset_dir.mkdir(exist_ok=True, parents=True)

    print("Loading dataset from torchgeo")
    cashew = BeninSmallHolderCashews(root=SRC_DATASET_DIR, download=True, checksum=True)

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(256, 256),
        n_time_steps=N_TIMESTEPS,
        bands_info=io.sentinel2_13_bands,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=LABEL_BAND,
        eval_loss=io.SegmentationAccuracy,  # TODO probably not the final loss eval loss. To be discussed.
        # either 50cm or 40cm, Airbus Pleiades 50cm, https://radiantearth.blob.core.windows.net/mlhub/technoserve-cashew-benin/Documentation.pdf
        spatial_resolution=SPATIAL_RESOLUTION,
    )
    task_specs.save(dataset_dir)

    multiprocess = True
    if multiprocess:
        with Pool(os.cpu_count()) as p:
            print(f"Spinning up pool of {os.cpu_count()} workers")
            iterator = p.imap(process_i, ((i, cashew_i, dataset_dir) for i, cashew_i in enumerate(cashew)))
            for i in tqdm(iterator, total=len(cashew)):
                pass

    else:
        for i, cashew_i in enumerate(tqdm(cashew)):
            process_i(i, cashew_i, dataset_dir)


if __name__ == "__main__":
    convert()
