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

import re
from typing import List
from ccb import io
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
import xmltodict
from warnings import warn
import csv

DATASET_NAME = "NeonTree"
SRC_DATASET_DIR = Path(io.src_datasets_dir, DATASET_NAME)
ZENODO_DATASET_DIR = Path(io.src_datasets_dir, DATASET_NAME + "_zenodo")

DATASET_DIR = Path(io.datasets_dir, DATASET_NAME)


def read_xml(xml_path):
    """parse the xml annotation file.

    Only the bounding box is extracted all other fields contain constant information except:
        * truncated: 30891 False, 152 True
        * difficult: 31012 False, 31 True
    , which doesn't seem useful.
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


def load_tif(tif_path):
    with rasterio.open(tif_path) as fd:
        data = fd.read()
        crs = fd.crs
        transform = fd.transform
        no_data = fd.nodata
    return np.moveaxis(data, 0, 2), crs, transform, no_data


def to_csv(info_list, dst_dir):
    with open(Path(dst_dir, "info.csv"), 'w') as fd:
        writer = csv.writer(fd)
        writer.writerows(info_list)


def find_missing(dir_list: List[Path], file_set):
    missing_list = []
    other_files = []
    for dir in dir_list:
        for file in dir.iterdir():
            if file.name.endswith('.tif'):
                if file not in file_set:
                    missing_list.append(file)
            else:
                other_files.append(file)
    print("Unused files from zenodo (training files):")
    for file in missing_list:
        print(file)


def _extract_tag(file_name):
    tags = re.findall('[A-Z]{4}', file_name)
    if len(tags) == 0:
        tag = None
    elif len(tags) == 1:
        tag = tags[0]
    else:
        print('more than one tag:', tags)
        tag = tags[0]
    return tag


def convert_dataset(src_dataset_dir, zenodo_dataset_dir, dataset_dir, max_count):
    sample_count = 0

    info_list = []
    file_set = set()

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
                split = 'test'
                sample_list = make_sample(name, rgb_path, chm_path, hs_path, boxes, check_shapes=True)

            elif np.all(exists[3:]):
                split = 'train'
                sample_list = make_sample(name, rgb_path_z, chm_path_z, hs_path_z, boxes, check_shapes=True, slice=True)
            else:
                split = 'unk'
                sample_list = []

            info = (name, tag, len(boxes), split) + tuple(exists)
            info_list.append(info)

            for sample in sample_list:
                partition.add(split, sample.sample_name)
                sample.write(dataset_dir)

                sample_count += 1
                if max_count is not None and sample_count >= max_count:
                    break

            if max_count is not None and sample_count >= max_count:
                break

    partition.save(dataset_dir, 'original')

    to_csv(info_list, dataset_dir)
    find_missing([zenodo_dataset_dir], file_set)


BAND_INFO_LIST = [
    io.SpectralBand("red"),
    io.SpectralBand("green"),
    io.SpectralBand("blue"),
    io.Height("Canopy Height Model", alt_names=("lidar", "CHM")),
    io.HyperSpectralBands("Neon", n_bands=369),
]


def extract_boxes(boxes, y_offset, x_offset, area_threshold=10):
    new_boxes = []

    def clip(box, key, offset):
        box[key] = int(np.clip(box[key] + offset, 0, 399))

    for box_ in boxes:
        box = box_.copy()
        clip(box, 'xmin', x_offset)
        clip(box, 'ymin', y_offset)
        clip(box, 'xmax', x_offset)
        clip(box, 'ymax', y_offset)

        area = (box['xmax'] - box['xmin']) * (box['ymax'] - box['ymin'])
        if area >= area_threshold:
            new_boxes.append(box)
    return new_boxes


def extract_slices(rgb_data, chm_data, hs_data, boxes, slice_shape):
    # TODO slice boxes
    def get_patch(data, start_x, start_y, scale=1):
        start_x, start_y, size_x, size_y = tuple(
            np.round(np.array([start_x, start_y, slice_shape[0], slice_shape[1]]) * scale).astype(np.int))
        return data[start_x:start_x + size_x, start_y: start_y + size_y, :]

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
            data_list.append((rgb_patch, chm_patch, hs_patch, new_boxes, f'_{i:02d}_{j:02d}'))
    return data_list


def make_sample(name, rgb_path, chm_path, hs_path, boxes, check_shapes=True, slice=False) -> io.Sample:

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
        data_list = [(rgb_data, chm_data, hs_data, boxes, '')]

    sample_list = []
    for rgb_data, chm_data, hs_data, new_boxes, suffix in data_list:
        for tag, data in (('rgb', rgb_data), ('chm', chm_data), ('hs', hs_data)):
            if np.any(data < 0):
                print(f'negative values in {tag}.')

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
                crs=crs)
            bands.append(band)

        bands.append(
            io.Band(
                chm_data, band_info=BAND_INFO_LIST[3],
                spatial_resolution=1, transform=chm_transform, crs=crs))
        bands.append(
            io.Band(
                hs_data, band_info=BAND_INFO_LIST[4],
                spatial_resolution=1, transform=hs_transform, crs=crs))

        sample_list.append(io.Sample(bands, label=new_boxes, sample_name=name + suffix))
    return sample_list


def convert(max_count=None, dataset_dir=DATASET_DIR):

    dataset_dir.mkdir(exist_ok=True, parents=True)

    task_specs = io.TaskSpecifications(
        dataset_name=DATASET_NAME,
        patch_size=(400, 400),
        n_time_steps=1,
        bands_info=BAND_INFO_LIST,
        bands_stats=None,  # Will be automatically written with the inspect script
        label_type=io.Detection(),
        eval_loss=io.Accuracy(),  # TODO what loss will we use?
        spatial_resolution=0.1,
    )
    task_specs.save(dataset_dir)

    convert_dataset(SRC_DATASET_DIR, ZENODO_DATASET_DIR, dataset_dir, max_count=max_count)


if __name__ == "__main__":

    convert()
