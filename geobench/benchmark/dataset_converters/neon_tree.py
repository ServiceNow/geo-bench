"""Neon Tree dataset."""
# Download using zenodo_get (or manual download: https://zenodo.org/record/4746605#.Yd7mtlhKgeb)
#
# For training tiles:
# $ pip install zenodo_get
# $ zenodo_get 5593238
#
# For Evaluation set:
# $ git clone https://github.com/weecology/NeonTreeEvaluation.git
#
# For running this code:
# $ pip install xmltodict

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union
from warnings import warn

import numpy as np
import rasterio
import xmltodict
from tqdm import tqdm

from geobench import io
from geobench.benchmark.rasterize_detection import rasterize_box

SEGMENTATION = True

if SEGMENTATION:
    DATASET_NAME = "NeonTree_segmentation"
else:
    DATASET_NAME = "NeonTree_detection"

SRC_DATASET_DIR = io.CCB_DIR / "source" / "NeonTree"  # type: ignore
# ZENODO_DATASET_DIR = Path(io.src_datasets_dir, DATASET_NAME + "_zenodo")
ZENODO_DATASET_DIR = SRC_DATASET_DIR / "_zenodo"  # type: ignore

DATASET_DIR = io.CCB_DIR / "converted" / DATASET_NAME


if SEGMENTATION:
    label_type = io.SegmentationClasses("label", spatial_resolution=0.1, n_classes=2, class_names=["no tree", "tree"])  # type: ignore
else:
    label_type = io.Detection()  # type: ignore


def read_xml(xml_path) -> List[Dict[str, int]]:
    """Parse the xml annotation file.

    Only the bounding box is extracted all other fields contain constant information except:
        * truncated: 30891 False, 152 True
        * difficult: 31012 False, 31 True
    , which doesn't seem useful.

    Args:
        xml_path: path to xml file

    Returns:
        bounding box annotations
    """
    with open(xml_path, "r") as fd:
        xml = fd.read()

    info = xmltodict.parse(xml)
    objects = info["annotation"]["object"]
    if not isinstance(objects, list):
        objects = [objects]

    boxes = []
    for object in objects:
        box = {key: int(val) for key, val in object["bndbox"].items()}
        boxes.append(box)

    return boxes


def load_tif(tif_path) -> Tuple["np.typing.NDArray[np.int_]", Any, Any, Any]:
    """Load tif file.

    Args:
        tif_path: path to tif file

    Returns:
        tif image data array
    """
    with rasterio.open(tif_path) as fd:
        data: "np.typing.NDArray[np.int_]" = fd.read()
        crs: Any = fd.crs
        transform: Any = fd.transform
        no_data: Any = fd.nodata
    return np.moveaxis(data, 0, 2), crs, transform, no_data


def to_csv(info_list: List[Tuple[Union[str, bool], ...]], dst_dir: str) -> None:
    """Save info to csv.

    Args:
        info_list: info to save
        dst_dir: path to directory where to save csv
    """
    with open(Path(dst_dir, "info.csv"), "w") as fd:
        writer = csv.writer(fd)
        writer.writerows(info_list)


def find_missing(dir_list: List[Path], file_set) -> None:
    """Find missing files.

    Args:
        dir_list: list of paths to directories
        file_set: set of current files
    """
    missing_list = []
    other_files = []
    for dir in dir_list:
        for file in dir.iterdir():
            if file.name.endswith(".tif"):
                if file not in file_set:
                    missing_list.append(file)
            else:
                other_files.append(file)
    print("Unused files from zenodo (training files):")
    for file in missing_list:
        print(file)


def _extract_tag(file_name: str) -> Union[None, str]:
    """Extract tag with regex.

    Args:
        file_name: file name

    Returns:
        tag if found
    """
    tags = re.findall("[A-Z]{4}", file_name)
    if len(tags) == 0:
        tag = None
    elif len(tags) == 1:
        tag = tags[0]
    else:
        print("more than one tag:", tags)
        tag = tags[0]
    return tag


def convert_dataset(src_dataset_dir: str, zenodo_dataset_dir: str, dataset_dir: str, max_count: int) -> None:
    """Convert dataset.

    Args:
        src_dataset_dir: source dataset directory
        zenodo_dataset_dir: directory to zenodo dataset
        dataset_dir: directory where to convert dataset to
        max_count: maximum number of samples
    """
    sample_count = 0

    info_list: List[Tuple[Union[str, bool], ...]] = []
    file_set: Set[Path] = set()

    partition = io.Partition()
    path_list = list(Path(src_dataset_dir, "annotations").iterdir())
    for label_path in tqdm(path_list):
        if label_path.suffix == ".xml":
            name = label_path.stem
            tag = _extract_tag(name)
            boxes = read_xml(label_path)

            rgb_path = Path(src_dataset_dir, "evaluation", "RGB", f"{name}.tif")
            hs_path = Path(src_dataset_dir, "evaluation", "Hyperspectral", f"{name}_hyperspectral.tif")
            chm_path = Path(src_dataset_dir, "evaluation", "CHM", f"{name}_CHM.tif")

            rgb_path_z = Path(zenodo_dataset_dir, f"{name}.tif")
            hs_path_z = Path(zenodo_dataset_dir, f"{name}_hyperspectral.tif")
            chm_path_z = Path(zenodo_dataset_dir, f"{name}_CHM.tif")

            all_paths = (rgb_path, hs_path, chm_path, rgb_path_z, hs_path_z, chm_path_z)
            exists = [p.exists() for p in all_paths]
            file_set.update(all_paths)

            # shapes = []
            # for p in all_paths:
            #     if p.exists():
            #         shapes.append(str(load_tif(p)[0].shape))
            #     else:
            #         shapes.append("None")

            if np.all(exists[:3]):
                split = "test"
                sample_list = make_sample(name, str(rgb_path), str(chm_path), str(hs_path), boxes, check_shapes=True)

            elif np.all(exists[3:]):
                split = "train"
                sample_list = make_sample(
                    name, str(rgb_path_z), str(chm_path_z), str(hs_path_z), boxes, check_shapes=True, slice=True
                )
            else:
                split = "unk"
                sample_list = []

            info = (str(name), str(tag), str(len(boxes)), str(split)) + tuple(exists)
            info_list.append(info)

            for sample in sample_list:
                partition.add(split, sample.sample_name)
                sample.write(dataset_dir)

                sample_count += 1
                if max_count is not None and sample_count >= max_count:
                    break

            if max_count is not None and sample_count >= max_count:
                break

    partition.resplit_iid(split_names=("valid", "test"), ratios=(0.5, 0.5))
    partition.save(dataset_dir, "original", as_default=True)

    to_csv(info_list, str(dataset_dir))
    find_missing([Path(zenodo_dataset_dir)], file_set)


BAND_INFO_LIST: List[Any] = io.make_rgb_bands(0.1)
BAND_INFO_LIST.append(io.ElevationBand("Canopy Height Model", alt_names=("lidar", "CHM"), spatial_resolution=0.1))
BAND_INFO_LIST.append(io.HyperSpectralBands("Neon", n_bands=369, spatial_resolution=1))


def extract_boxes(boxes, y_offset, x_offset, area_threshold=10) -> List[Dict[str, int]]:
    """Extract bounding boxes.

    Args:
        boxes: list of bounding boxes
        y_offset: y offset for box
        x_offset: x offset for box
        area_threshold: minimum area of bounding box to extract

    Return:
        extracted bounding boxes
    """
    new_boxes = []

    def clip(box, key, offset):
        box[key] = int(np.clip(box[key] + offset, 0, 399))

    for box_ in boxes:
        box = box_.copy()
        clip(box, "xmin", x_offset)
        clip(box, "ymin", y_offset)
        clip(box, "xmax", x_offset)
        clip(box, "ymax", y_offset)

        area = (box["xmax"] - box["xmin"]) * (box["ymax"] - box["ymin"])
        if area >= area_threshold:
            new_boxes.append(box)
    return new_boxes


def extract_slices(rgb_data, chm_data, hs_data, boxes, slice_shape):
    """Extract image patch slices.

    Args:
        rgb_data: RGB imagery data
        chm_data: canopy height model data
        hs_data: hyperspectral data
        boxes: bounding boxes
        slice_shape: desired shape of slice

    Returns:
        sliced data
    """
    # TODO slice boxes
    def get_patch(data, start_x, start_y, scale=1):
        start_x, start_y, size_x, size_y = tuple(
            np.round(np.array([start_x, start_y, slice_shape[0], slice_shape[1]]) * scale).astype(np.int)
        )
        return data[start_x : start_x + size_x, start_y : start_y + size_y, :]

    shape = np.array(rgb_data.shape[:2])
    slice_shape = np.asarray(slice_shape)
    n_tiles = np.ceil(shape / slice_shape).astype(np.int)
    stride = np.floor((shape - slice_shape) / (n_tiles - 1)).astype(np.int)
    n_x, n_y = tuple(n_tiles)
    stride_x, stride_y = tuple(stride)
    data_list = []
    for j in range(n_y):
        for i in range(n_x):
            rgb_patch = get_patch(rgb_data, stride_x * i, stride_y * j, scale=1)
            chm_patch = get_patch(chm_data, stride_x * i, stride_y * j, scale=0.1)
            hs_patch = get_patch(hs_data, stride_x * i, stride_y * j, scale=0.1)
            new_boxes = extract_boxes(boxes, -stride_x * i, -stride_y * j)
            data_list.append((rgb_patch, chm_patch, hs_patch, new_boxes, f"_{i:02d}_{j:02d}"))
    return data_list


def make_sample(
    name: str, rgb_path: str, chm_path: str, hs_path: str, boxes, check_shapes: bool = True, slice: bool = False
) -> List[io.Sample]:
    """Create a sample.

    Args:
        name: name of sample
        rgb_path: path to rgb data
        chm_path: path to canopy height model data
        hs_path: path to hyperspectral data
        boxes: set of bounding boxes
        check_shapes: whether or not to check shapes before making sample
        slice: whether or not to slice sample

    Returns:
        sample
    """
    rgb_data, crs, rgb_transform, rgb_nodata = load_tif(rgb_path)
    chm_data, chm_crs, chm_transform, chm_nodata = load_tif(chm_path)
    hs_data, _, hs_transform, hs_nodata = load_tif(hs_path)
    assert crs == chm_crs

    if hs_data.shape[2] == 426:
        hs_data = hs_data[:, :, :369]  # TODO fix to the right set of bands

    # TODO fix Temporary hack for the nodata
    chm_data[chm_data == chm_nodata] = 0
    hs_data[hs_data == chm_nodata] = 0

    if slice:
        data_list = extract_slices(rgb_data, chm_data, hs_data, boxes, slice_shape=(400, 400))
    else:
        data_list = [(rgb_data, chm_data, hs_data, boxes, "")]

    sample_list: List[io.Sample] = []
    for rgb_data, chm_data, hs_data, new_boxes, suffix in data_list:
        for tag, data in (("rgb", rgb_data), ("chm", chm_data), ("hs", hs_data)):
            if np.any(data < 0):
                print(f"negative values in {tag}.")

        if check_shapes:
            shapes = (rgb_data.shape, chm_data.shape, hs_data.shape)
            target_shapes = ((400, 400, 3), (40, 40, 1), (40, 40, 369))
            if shapes != target_shapes:
                warn(f"skipping {name}, shapes (rgb, chm, hyperspectral) = {shapes} != {target_shapes}")
                return sample_list

        bands = []
        for i in range(3):
            band = io.Band(
                data=rgb_data[:, :, i],
                band_info=BAND_INFO_LIST[i],
                spatial_resolution=0.1,
                transform=rgb_transform,
                crs=crs,
            )
            bands.append(band)

        bands.append(
            io.Band(chm_data, band_info=BAND_INFO_LIST[3], spatial_resolution=1, transform=chm_transform, crs=crs)
        )
        bands.append(
            io.Band(hs_data, band_info=BAND_INFO_LIST[4], spatial_resolution=1, transform=hs_transform, crs=crs)
        )

        if SEGMENTATION:
            label_data = rasterize_box(boxes=new_boxes, img_shape=rgb_data.shape[:2], scale=0.6)
            label = io.Band(
                data=label_data, band_info=label_type, spatial_resolution=0.1, transform=rgb_transform, crs=crs
            )
        else:
            label = new_boxes
        sample_list.append(io.Sample(bands, label=label, sample_name=name + suffix))
    return sample_list


def convert(max_count=None, dataset_dir=DATASET_DIR) -> None:
    """Convert Neon Tree dataset dataset.

    Args:
        max_count: maximum number of samples
        dataset_dir: path to dataset directory
    """
    dataset_dir.mkdir(exist_ok=True, parents=True)

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(400, 400),
        n_time_steps=1,
        bands_info=BAND_INFO_LIST,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=label_type,
        eval_loss=io.Accuracy(),  # TODO what loss will we use?
        spatial_resolution=0.1,
    )
    task_specs.save(dataset_dir, overwrite=True)

    convert_dataset(str(SRC_DATASET_DIR), str(ZENODO_DATASET_DIR), str(dataset_dir), max_count=max_count)


if __name__ == "__main__":

    convert()
