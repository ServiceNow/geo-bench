# EuroSat will be automatically downloaded by TorchGeo (https://github.com/microsoft/torchgeo)


from ccb import io
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchgeo.datasets import EuroSAT

DATASET_NAME = "eurosat"
SRC_DATASET_DIR = Path(io.src_datasets_dir, DATASET_NAME)
DATASET_DIR = Path(io.datasets_dir, DATASET_NAME)


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


def convert(max_count=None, dataset_dir=DATASET_DIR):
    dataset_dir.mkdir(exist_ok=True, parents=True)

    eurosat_dataset = EuroSAT(root=SRC_DATASET_DIR, split="train", transforms=None, download=True, checksum=True)

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(64, 64),
        n_time_steps=1,
        bands_info=io.sentinel2_13_bands,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=io.Classification(10),
        eval_loss=io.Accuracy,
        spatial_resolution=10,
    )
    task_specs.save(dataset_dir)

    for i, tg_sample in enumerate(tqdm(eurosat_dataset)):
        sample_name = f"id_{i:04d}"

        images = np.array(tg_sample["image"])
        label = tg_sample["label"]

        sample = make_sample(images, int(label), sample_name)
        sample.write(dataset_dir)

        # temporary for creating small datasets for development purpose
        if max_count is not None and i + 1 >= max_count:
            break


if __name__ == "__main__":
    convert()