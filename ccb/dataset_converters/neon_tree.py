# Download using zenodo_get (or manual download: https://zenodo.org/record/4746605#.Yd7mtlhKgeb)
# $ pip install zenodo_get
# $ zenodo_get 4746605
# $ git clone https://github.com/weecology/NeonTreeEvaluation.git
# pip install xmltodict

from ccb import io
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
import xmltodict
from warnings import warn

DATASET_NAME = "NeonTree"
SRC_DATASET_DIR = Path(io.src_datasets_dir, DATASET_NAME)
DATASET_DIR = Path(io.datasets_dir, DATASET_NAME)


def read_xml(xml_path):
    """parse the xml annotation file.

    Only the bounding box is extracted all other fields contain redundant information except:
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
    return np.moveaxis(data, 0, 2), crs, transform


def convert_dataset(src_dataset_dir, dataset_dir, max_count):
    sample_count = 0
    for label_path in Path(src_dataset_dir, "annotations").iterdir():
        if label_path.suffix == ".xml":
            name = label_path.stem
            boxes = read_xml(label_path)

            rgb_path = Path(src_dataset_dir, "evaluation", "RGB", f"{name}.tif")
            hs_path = Path(src_dataset_dir, "evaluation", "Hyperspectral", f"{name}_hyperspectral.tif")
            chm_path = Path(src_dataset_dir, "evaluation", "CHM", f"{name}_CHM.tif")

            exists = [p.exists() for p in (rgb_path, hs_path, chm_path)]

            if not np.all(exists):
                print(f"Skipping {name}. Exists: {exists}")
                continue

            make_sample(name, rgb_path, chm_path, hs_path, boxes, dataset_dir)

            sample_count += 1
            if max_count is not None and sample_count >= max_count:
                break
        else:
            print(f"Unknown file {label_path}.")


BAND_INFO_LIST = [
    io.SpectralBand("red"),
    io.SpectralBand("green"),
    io.SpectralBand("blue"),
    io.Height("Canopy Height Model", alt_names=("lidar", "CHM")),
    io.HyperSpectralBands("Neon", n_bands=426),
]


def make_sample(name, rgb_path, chm_path, hs_path, boxes, dataset_dir) -> io.Sample:

    rgb_data, crs, rgb_transform = load_tif(rgb_path)
    chm_data, chm_crs, chm_transform = load_tif(chm_path)
    hs_data, _, hs_transform = load_tif(hs_path)
    assert crs == chm_crs

    shapes = (rgb_data.shape, chm_data.shape, hs_data.shape)
    target_shapes = ((400, 400, 3), (40, 40, 1), (40, 40, 426))
    if shapes != target_shapes:
        warn(f"skipping {name}, shapes (rgb, chm, hyperspectral) = {shapes} != {target_shapes}")
        return

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

    sample = io.Sample(bands, label=boxes, sample_name=name)
    sample.write(dataset_dir)


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

    convert_dataset(SRC_DATASET_DIR, dataset_dir, max_count=max_count)


if __name__ == "__main__":

    convert()
