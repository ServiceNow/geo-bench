# Xview2 will be automatically downloaded by TorchGeo (https://github.com/microsoft/torchgeo)
# https://torchgeo.readthedocs.io/en/latest/api/datasets.html#xview2

# Need to download dataset manually...
# Register at https://xview2.org/signup
# Copy link address to download Challenge training set (~7.8GB) and test set (~2.6GB) from https://xview2.org/download-links
# Insert link address to the curl to download datasets:
# curl -o $CC_BENCHMARK_SOURCE_DATASETS/xview2/train_images_labels_targets.tar.gz --remote-name "https://download.xview2.org/train_images_labels_targets.tar.gz?Expires=<>&Signature=<>__&Key-Pair-Id=<>"
# curl -o $CC_BENCHMARK_SOURCE_DATASETS/xview2/test_images_labels_targets.tar.gz --remote-name "INSERT_TEST_DATA_LINK_HERE"
# Verify download by visually comparing SHASUMs
# shasum -a 1 train_images_labels_targets.tar.gz
# shasum -a 1 test_images_labels_targets.tar.gz
# 

from ccb import io
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchgeo.datasets import XView2

DATASET_NAME = "xview2"
SRC_DATASET_DIR = Path(io.src_datasets_dir, DATASET_NAME)
DATASET_DIR = Path(io.datasets_dir, DATASET_NAME)

# Todo: move to io.dataset.py
class Worldview3(io.SpectralBand):
    "Visual band from Maxar Digitalglobe Worldview 3"

# Worldview3 
# Source: https://www.spaceimagingme.com/downloads/sensors/
#      datasheets/DG_WorldView3_DS_2014.pdf
# Todo: verify that band ordering is BGR
worldview3_rgb_bands = [
    Worldview3("01 - Blue", ("1", "01", "blue"), 
             spatial_resolution=1.24, 
             wavelength=0.51), # .45-.51 
    Worldview3("02 - Green", ("2", "02", "green"), 1.24, 0.58), # .51-.58
    Worldview3("03 - Red", ("3", "03", "red"), 1.24, 0.69), # .63-.69
]

# Todo: document class labels: background, no damage, minor damage, major damage, destroyed
LABEL_BAND = io.SegmentationClasses("label", spatial_resolution=1.24, n_classes=5)

def make_sample(images, mask, sample_name):
    n_bands, _height, _width = images.shape

    # TODO: Get geotransfrom from torchgeo
    # Todo: Get date, resolution, and meta_info from image
    # Todo: Convert date from "capture_date": "2018-02-05T17:10:18.000Z", 
    #    to Union[datetime.datetime, datetime.date]
    transform = None  
    crs = None
    date = None
    # Note: Spatial resolution varies from 1.24 to 1.38m. 
    spatial_resolution = 1.24 # images['gsd']
    meta_info = None
    """ = {"off_nadir_angle": images['off_nadir_angle'], 
                 "pan_resolution": images['pan_resolution'],
                 "sun_azimuth": images['sun_azimuth'],
                 "sun_elevation": images['sun_elevation'],
                 "target_azimuth": images['target_azimuth']}
    """
    bands = []
    for band_idx in range(n_bands):
        band_data = images[band_idx, :, :]

        band_info = worldview3_rgb_bands[band_idx]
        
        # Todo: where to store spatial_resolution that varies per image?
        band = io.Band(
            data=band_data,
            band_info=band_info,
            spatial_resolution=spatial_resolution, 
            date=date,
            transform=transform,
            crs=crs,
            meta_info=meta_info,
            convert_to_int16=False,
        )
        bands.append(band)

    # Todo: Is it okay if mask has shape: (height, width) with class_id = value or 
    #  Should mask have shape (height, width, classes) 
    label = io.Band(data=mask, band_info=LABEL_BAND, 
                    spatial_resolution=spatial_resolution, 
                    transform=transform, crs=crs)
    return io.Sample(bands, label=label, sample_name=sample_name)


def convert(max_count=None, dataset_dir=DATASET_DIR):
    """
    Converts torchgeo.XView2 dataset into ccb dataset 
    Args:
        max_count int: Maximum number of images to be converted
        dataset_dir string: Dataset directory
    Returns:
    """
    dataset_dir.mkdir(exist_ok=True, parents=True) # Creates path to converted data
    partition = io.dataset.Partition() # Creates dictionary to store train, val, test filenames

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(1024, 1024),
        n_time_steps=1,
        bands_info=worldview3_rgb_bands,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=LABEL_BAND,
        eval_loss=io.SegmentationAccuracy,  # TODO probably not the final 
                                            # loss eval loss. To be discussed.
        spatial_resolution=1.24,
    )
    task_specs.save(dataset_dir)

    offset = 0
    import pdb;pdb.set_trace()

    for split_name in ["train", "val", "test"]:
        xview2_dataset = XView2(root=SRC_DATASET_DIR, split=split_name, 
                                transforms=None, checksum=True)
        for i, tg_sample in enumerate(tqdm(xview2_dataset)):
            # TODO: How to incorporate change detection dataset that contains pre and post image?
            for j, pair_id in enumerate(['_pre', '_post']):
                sample_name = f"id{pair_id:s}_{i+offset:04d}"

                images = np.array(tg_sample["image"][j,...])
                mask = tg_sample["mask"][j,...]

                sample = make_sample(images, mask, sample_name)
                sample.write(dataset_dir)
                partition.add(split_name.replace("val", "valid"), sample_name)

                offset += 1

                # temporary for creating small datasets for development purpose
                if max_count is not None and i + 1 >= max_count:
                    break

        # temporary for creating small datasets for development purpose
        if max_count is not None and offset >= max_count:
            break

    partition.save(dataset_dir, "original")


if __name__ == "__main__":
    convert(2)
