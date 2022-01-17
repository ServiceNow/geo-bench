# Smallholder Cashew Dataset will be downloaded by torchgeo
# This requires Radiant MLHub package and API token
# pip install radiant_mlhub
# More info on the dataset: https://mlhub.earth/10.34911/rdnt.hfv20i

from ccb.dataset import io
import numpy as np
from dataset_converters import util
from pathlib import Path
from tqdm import tqdm
from toolbox.core.task_specs import TaskSpecifications
from toolbox.core.loss import SegmentationAccuracy
from torchgeo.datasets import BeninSmallHolderCashews


# Classification labels
LABELS = (
    'no data',
    'well-managed plantation',
    'poorly-managed plantation',
    'non-plantation',
    'residential',
    'background',
    'uncertain'
)
SPATIAL_RESOLUTION = 0.5  # meters, to be confirmed
N_TIMESTEPS = 70
LABEL_BAND = io.SegmentationClasses("label", spatial_resolution=SPATIAL_RESOLUTION, n_classes=len(LABELS))

# Paths
src_dataset_dir = Path(util.src_datasets_dir, "smallholder_cashew")
dataset_dir = Path(util.dst_datasets_dir, "smallholder_cashew")
dataset_dir.mkdir(exist_ok=True, parents=True)

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
                spatial_resolution=SPATIAL_RESOLUTION,
                transform=transform,
                crs=crs,
                convert_to_int16=False,
            )
            bands.append(band)

    label = io.Band(data=mask, band_info=LABEL_BAND, 
        spatial_resolution=SPATIAL_RESOLUTION, transform=transform, crs=crs)
    return io.Sample(bands, label=label, sample_name=sample_name)



if __name__ == "__main__":

    print('Loading dataset from torchgeo')
    cashew = BeninSmallHolderCashews(root=src_dataset_dir, download=True, checksum=True)

    task_specs = TaskSpecifications(
        dataset_name="BeninSmallHolderCashews",
        patch_size=(256, 256),
        n_time_steps=N_TIMESTEPS,
        bands_info=io.sentinel2_13_bands,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=LABEL_BAND,
        eval_loss=SegmentationAccuracy,  # TODO probably not the final loss eval loss. To be discussed.
        spatial_resolution=SPATIAL_RESOLUTION,  # either 50cm or 40cm, Airbus Pleiades 50cm, https://radiantearth.blob.core.windows.net/mlhub/technoserve-cashew-benin/Documentation.pdf
    )
    task_specs.save(dataset_dir)

    for i, tg_sample in enumerate(tqdm(cashew)):
        sample_name = f"sample={i:04d}"

        images = np.array(tg_sample["image"])
        mask = tg_sample["mask"]

        sample = make_sample(images, mask, sample_name)
        sample.write(dataset_dir)

        # temporary for creating small datasets for development purpose
        if i > 100:
            break

