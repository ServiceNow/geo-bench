# Chesapeake Bay Land Cover dataset will be automatically downloaded by
# TorchGeo (https://github.com/microsoft/torchgeo)

from ccb import io
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torchgeo.datamodules import ChesapeakeCVPRDataModule

# Note: both of the following datasets need to be downloaded manually
# into the same directory. It will not download successfully using
# download=True as in other datasets from torchgeo. See Github issue:
# https://github.com/microsoft/torchgeo/issues/452#issuecomment-1059469588
# 1. Primary dataset: https://lila.science/datasets/chesapeakelandcover
#     (use azcopy)
# 2. Extension: https://zenodo.org/record/5866525#.YlhpH27MJf0

DATASET_NAME = "cvpr_chesapeake_landcover"
SRC_DATASET_DIR = Path(io.src_datasets_dir, DATASET_NAME)
DATASET_DIR = Path(io.datasets_dir, DATASET_NAME)
# See dataset documentation for more details on below:
# https://torchgeo.readthedocs.io/en/latest/api/datasets.html#torchgeo.datasets.ChesapeakeCVPR
SPATIAL_RESOLUTION = 1  # meters
PATCH_SIZE = 256
# Classification labels
LABELS = (
    "water",
    "tree-canopy-forest",
    "low-vegetation-field",
    "barren-land",
    "impervious-other",
    "impervious-roads",
    "no-data",
)
LABEL_BAND = io.SegmentationClasses("label", spatial_resolution=SPATIAL_RESOLUTION, n_classes=len(LABELS))

BAND_INFO_LIST = io.make_rgb_bands(SPATIAL_RESOLUTION)
BAND_INFO_LIST.append(io.SpectralBand("NearInfrared", ("nir",), SPATIAL_RESOLUTION, 0.876))


def make_sample(image, label, sample_name, task_specs, crs):
    n_bands, _height, _width = image.shape

    if (_height, _width) != (PATCH_SIZE, PATCH_SIZE):
        image = image[:, :PATCH_SIZE, :PATCH_SIZE]
        n_bands, _height, _width = image.shape

    transform = None  # TODO can't find the GPS coordinates from torch geo.

    bands = []
    for band_idx in range(n_bands):
        band_data = image[band_idx, :, :]

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

    label = io.Band(
        data=label, band_info=LABEL_BAND, spatial_resolution=SPATIAL_RESOLUTION, transform=transform, crs=crs
    )

    return io.Sample(bands, label=label, sample_name=sample_name)


def convert(max_count=None, dataset_dir=DATASET_DIR):
    dataset_dir.mkdir(exist_ok=True, parents=True)
    np.random.seed(0)  # Set random seed for reproducibility
    partition = io.dataset.Partition()

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(PATCH_SIZE, PATCH_SIZE),
        n_time_steps=1,
        bands_info=BAND_INFO_LIST,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=LABEL_BAND,
        eval_loss=io.SegmentationAccuracy,
        spatial_resolution=SPATIAL_RESOLUTION,
    )
    task_specs.save(dataset_dir, overwrite=True)

    states = ["de", "md", "va", "wv", "pa", "ny"]

    dm = ChesapeakeCVPRDataModule(
        root_dir=SRC_DATASET_DIR,
        train_splits=[f"{state}-train" for state in states],
        val_splits=[f"{state}-val" for state in states],
        test_splits=[f"{state}-test" for state in states],
        patches_per_tile=500,
        patch_size=PATCH_SIZE,
        batch_size=1,
        num_workers=0,
        class_set=len(LABELS),
    )

    dm.prepare_data()
    dm.setup()

    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()

    n_samples = 0
    for s_idx, split_dl in enumerate([train_dl, val_dl, test_dl]):
        for i, dl_sample in enumerate(tqdm(split_dl)):
            sample_name = f"id_{n_samples:06d}"
            image = np.array(dl_sample["image"])[0]
            label = np.array(dl_sample["mask"])[0]
            crs = dl_sample["crs"][0]

            sample = make_sample(image, label, sample_name, task_specs, crs)
            sample.write(dataset_dir)

            if s_idx == 0:
                partition.add("train", sample_name)
            elif s_idx == 1:
                partition.add("valid", sample_name)
            elif s_idx == 2:
                partition.add("test", sample_name)

            n_samples += 1
            if max_count is not None and n_samples >= max_count:
                break

        if max_count is not None and n_samples >= max_count:
            break
    partition.save(dataset_dir, "original", as_default=True)


if __name__ == "__main__":
    convert()
