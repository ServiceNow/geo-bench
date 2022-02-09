# So2Sat will be automatically downloaded by TorchGeo (https://github.com/microsoft/torchgeo)


from ccb import io
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchgeo.datasets import So2Sat

DATASET_NAME = "so2sat"
SRC_DATASET_DIR = Path(io.src_datasets_dir, DATASET_NAME)
DATASET_DIR = Path(io.datasets_dir, DATASET_NAME)


def make_sample(images, label, sample_name, task_specs):
    n_bands, _height, _width = images.shape

    transform = None  # TODO can't find the GPS coordinates from torch geo.
    crs = None

    bands = []
    for band_idx in range(n_bands):
        band_data = images[band_idx, :, :]

        band_info = task_specs.bands_info[band_idx]
        band_data = band_data.astype(np.float32)
        band = io.Band(
            data=band_data,
            band_info=band_info,
            spatial_resolution=task_specs.spatial_resolution,
            transform=transform,
            crs=crs,
            convert_to_int16=False,
        )
        bands.append(band)

    return io.Sample(bands, label=label, sample_name=sample_name)


def convert(max_count=None, dataset_dir=DATASET_DIR):
    dataset_dir.mkdir(exist_ok=True, parents=True)
    partition = io.dataset.Partition()

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(32, 32),
        n_time_steps=1,
        bands_info=io.sentinel1_8_bands + io.sentinel2_13_bands[1:9] + io.sentinel2_13_bands[-2:],
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=io.Classification(17),
        eval_loss=io.Accuracy,
        spatial_resolution=10,
    )
    task_specs.save(dataset_dir)
    n_samples = 0
    for split_name in ["train", "validation", "test"]:
        so2sat_dataset = So2Sat(root=SRC_DATASET_DIR, split=split_name, transforms=None, checksum=True)
        for i, tg_sample in enumerate(tqdm(so2sat_dataset)):
            sample_name = f"id_{i:04d}"

            images = np.array(tg_sample["image"])
            label = tg_sample["label"]

            sample = make_sample(images, int(label), sample_name, task_specs)
            sample.write(dataset_dir)

            partition.add(split_name.replace("validation", "valid"), sample_name)

            n_samples += 1
            if max_count is not None and n_samples >= max_count:
                break

        if max_count is not None and n_samples >= max_count:
            break

    partition.save(dataset_dir, "original")


if __name__ == "__main__":
    convert(200)
