# Downloaded from "https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg13/brick_kiln.html"
# Try this command for downloading on headless server:
#   wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1aOWHRY72LlHNv7nwbAcPEHWcuuYnaEtx' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1aOWHRY72LlHNv7nwbAcPEHWcuuYnaEtx" -O brick_kiln_v1.0.tar.gz && rm -rf /tmp/cookies.txt


from ccb.dataset import io
import numpy as np
import csv
from dataset_converters import util
import rasterio
from pathlib import Path
import h5py
from tqdm import tqdm
from toolbox.core.task_specs import TaskSpecifications
from toolbox.core.loss import Accuracy
from torchgeo.datasets import EuroSAT
from multiprocessing import Pool

src_dataset_dir = Path(util.src_datasets_dir, "eurosat")
dataset_dir = Path(util.dst_datasets_dir, "eurosat")
dataset_dir.mkdir(exist_ok=True, parents=True)


def make_sample(images, label, sample_name):
    n_bands, _height, _width = images.shape

    transform = None  # TODO can't find the GPS coordinates from torch geo.
    crs = None

    bands = []
    for band_idx in range(n_bands):
        band_data = images[band_idx, :, :]

        band_info = io.sentinel2_13_bands[band_idx]

        band = io.Band(
            data=band_data,
            band_info=band_info,
            spatial_resolution=10,
            transform=transform,
            crs=crs,
            convert_to_int16=False,
        )
        bands.append(band)

    # label = io.Band(data=mask, band_info=LABEL_BAND, spatial_resolution=10, transform=transform, crs=crs)
    return io.Sample(bands, label=label, sample_name=sample_name)


if __name__ == "__main__":

    eurosat_dataset = EuroSAT(root=src_dataset_dir, split="train", transforms=None, download=True, checksum=True)

    task_specs = TaskSpecifications(
        dataset_name="EuroSAT",
        patch_size=(64, 64),
        n_time_steps=1,
        bands_info=io.sentinel2_13_bands,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=io.Classification(10),
        eval_loss=Accuracy,
        spatial_resolution=10,
    )
    task_specs.save(dataset_dir)

    for i, tg_sample in enumerate(tqdm(eurosat_dataset)):
        sample_name = f"sample={i:04d}"

        images = np.array(tg_sample["image"])
        label = tg_sample["label"]

        sample = make_sample(images, int(label), sample_name)
        sample.save_sample(dataset_dir)

        # temporary for creating small datasets for development purpose
        if i > 100:
            break
