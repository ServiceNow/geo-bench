import json
import os
import re
from functools import reduce
from typing import Dict, List, Tuple, Optional

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from collections import defaultdict

def percentile_normalization(
    img: np.array,
    lower: float = 2,
    upper: float = 98,
) -> np.array:
    """Applies percentile normalization to an input image.

    Args:
        img: image to normalize
        lower: lower percentile in range [0,100]
        upper: upper percentile in range [0,100]

    Returns
        normalized version of ``img``
    """
    assert lower < upper
    lower_percentile = np.percentile(img, lower)
    upper_percentile = np.percentile(img, upper)
    img_normalized = np.clip(
        (img - lower_percentile) / (upper_percentile - lower_percentile), 0, 1
    )
    return img_normalized

def create_patches(
    imgs: np.array,
    mask: np.array,
    patch_size: Tuple[
        int,
    ],
    label_threshold: float,
) -> List[Tuple[np.array,]]:
    """From loaded images create patches of desired size.

    Args:
        imgs: array of images T x C x H x W for corresponding mask
        mask: mask to time-series image array
        patch_size: desired size of patches
        label_threshold: patches' area must be filled with this percentage of labels

    Returns:
        patches as tuple of (image, mask)
    """
    shape = np.array(imgs.shape[2:])
    patch_shape = np.asarray(patch_size)
    n_tiles = np.ceil(shape / patch_shape).astype(np.int)
    stride = np.floor((shape - patch_shape) / (n_tiles - 1)).astype(np.int)
    n_x, n_y = tuple(n_tiles)
    stride_x, stride_y = tuple(stride)
    patches = []
    for j in range(n_y):
        for i in range(n_x):
            img_patch = get_patch(imgs, stride_x * i, stride_y * j, patch_shape, is_mask=False)
            mask_patch = get_patch(mask, stride_x * i, stride_y * j, patch_shape, is_mask=True)
            # apply threshold
            if compute_area_with_labels(mask_patch) >= label_threshold:
                patches.append((img_patch, mask_patch))
            else:
                continue

    return patches


def get_patch(
    data: np.array,
    start_x: int,
    start_y: int,
    patch_shape: Tuple[
        int,
    ],
    is_mask: bool = False,
) -> np.array:
    """Extract a patch from either the time-series images or the mask.

    Args:
        data: images or mask array
        start_x: index x
        start_y: index y
        patch_shape: shape of desired patch
        is_mask: whether to extract patch from mask or images

    Returns:
        Extracted patch of desired shape
    """
    start_x, start_y, size_x, size_y = tuple(
        np.round(np.array([start_x, start_y, patch_shape[0], patch_shape[1]])).astype(np.int16)
    )
    if is_mask:
        return data[start_x : start_x + size_x, start_y : start_y + size_y]
    else:
        return data[:, :, start_x : start_x + size_x, start_y : start_y + size_y]


def compute_area_with_labels(mask: np.array) -> float:
    """Compute percentage of mask that contain labels.

    Args:
        mask: mask to compute labels on

    Return:
        percentage
    """
    num_px_with_label = np.count_nonzero(mask)
    num_px_total = reduce(lambda x, y: x * y, list(mask.shape))
    return num_px_with_label / num_px_total


def collect_dates(filepaths: List[str], regex: re) -> List[str]:
    """Collect dates from images.

    Args:
        filepaths: paths of image directories that contain date
        regex: regex to match and collect date

    Returns:
        string of date in format %Y%m%d

    """
    dates = []
    for path in filepaths:
        match = re.search(regex, path)
        year = match.group("year")
        month = match.group("month")
        day = match.group("day")
        dates.append(str(year) + str(month) + str(day))

    return dates


def load_image_bands(filepath: str, bandnames: List[str], dest_crs: CRS) -> np.array:
    """Load seperate band images.

    Args:
        filepaths: one por more files to load and merge
        bandnames: band names with file extension as they can be found in data directory
        dest_crs: CRS of data in partition that is being created

    Returns:
        images at this filepath of shape C x H x W
    """
    # load imagery
    band_list = []
    for bandname in bandnames:
        band_filename = os.path.join(filepath, bandname)
        src = load_warp_file(band_filename, dest_crs)
        band = src.read()
        band_list.append(band)

    # stack along band channel dimension
    data = np.concatenate(band_list, axis=0, dtype=np.int16)
    return data


def load_warp_file(filepath: str, dest_crs: CRS) -> DatasetReader:
    """Load and warp a file to the correct CRS and resolution.

    Args:
        filepath: file to load and warp
        dest_crs: CRS of data in partition that is being created

    Returns:
        file handle of warped VRT
    """
    src = rasterio.open(filepath)

    # Only warp if necessary
    if src.crs != dest_crs:
        vrt = WarpedVRT(src, crs=dest_crs)
        src.close()
        return vrt
    else:
        return src


def load_geojson_mask(
    filepath: str,
    crop_type_key: str,
    height: int,
    width: int,
    class2idx: Dict[str, int],
    bounds: Dict[str, float],
) -> np.array:
    """Load the mask.

    Args:
        filepath: filepath to label
        crop_type_key: key in geojson file to find label
        height: of image to rasterize corresponding label
        width: of image to rasterize corresponding label
        class2idx: mapping of label class to numerical class
        bounds: image boundaries in CRS

    Returns:
        mask at this filepath
    """
    # only images are patched into tiles, masks are large, so remove duplicates
    per_label_shapes= defaultdict(list)
    with open(os.path.join(filepath)) as f:
        data = json.load(f)

    for feature in data["features"]:
        label = feature["properties"][crop_type_key]
        per_label_shapes[label].append(feature["geometry"])
    
    transform = rasterio.transform.from_bounds(
        bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"], width, height
    )
    if per_label_shapes:
        # create a mask tensor per class that is being found and then concatenate
        # to single segmentation mask with labels
        mask_list = []
        for label, shapes in per_label_shapes.items():

            label_mask = rasterize(shapes, out_shape=(int(height), int(width)), transform=transform)

            # assign correct segmentation label
            label_mask *= class2idx[label]
            mask_list.append(label_mask)

        mask_stack = np.stack(mask_list, axis=0)
         # assumes non-overlapping labels
        mask = np.max(mask_stack, axis=0, dtype=np.int16)
    else:
        mask = np.zeros((height, width))
   
    return mask


def load_tif_mask(
    filepath: str,
    dest_crs: CRS,
) -> np.array:
    """Load the mask.

    Args:
        filepaths: one or more files to load and merge
        query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

    Returns:
        labels at this query index
    """
    # load label
    label_filename = os.path.join(filepath, "labels.tif")

    src = load_warp_file(label_filename, dest_crs=dest_crs)
    label = src.read().astype(np.int16)

    return label
