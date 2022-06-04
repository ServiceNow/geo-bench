from matplotlib.pyplot import cla
import pytest
import numpy as np
import tempfile
from rasterio import features as rasterio_features
import json
import os
from rasterio.crs import CRS
from rasterio.transform import Affine
import rasterio
from ccb.dataset_converters.crop_type_utils import load_geojson_mask


def test_load_geojson_mask():
    """Test loading of geojson mask."""

    SHAPE = (10, 10)

    input_mask = np.zeros(SHAPE, dtype=np.int16)

    crop_labels = ["No Data", "Sunflower"]

    CLASS2IDX = {c: i for i, c in enumerate(crop_labels)}

    # turn corners into mask
    input_mask[0:2, 0:2] = 1
    input_mask[-2:, -2:] = 1
    input_mask[0:2, -2:] = 1
    input_mask[-2:, 0:2] = 1

    transform = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    # turn into geosjon
    labels = []
    for shape, value in rasterio_features.shapes(input_mask, transform=transform):
        if value == 1:
            label = {"type": "Feature", "properties": {"Crop": "Sunflower"}}
            label["geometry"] = shape
            labels.append(label)

    with tempfile.TemporaryDirectory() as tmpdirname:
        data_root = tmpdirname

        # create the geojson of proper format
        geojson = {"type": "FeatureCollection", "features": labels}

        # temp file of geojson
        geojson_filepath = os.path.join(data_root, "labels.geojson")
        with open(geojson_filepath, "w") as f:
            json.dump(geojson, f)

        # create corresponding image
        img = np.random.random((3, SHAPE[0], SHAPE[1]))
        img_filepath = os.path.join(data_root, "raster.tif")
        with rasterio.open(
            img_filepath,
            "w",
            driver="GTiff",
            height=img.shape[1],
            width=img.shape[2],
            count=img.shape[0],
            dtype=img.dtype,
            crs="epsg:4326",
            transform=transform,
            compress="zstd",
            predictor=2,
        ) as dst:

            dst.write(img)

        output_mask = load_geojson_mask(
            img_filepath=img_filepath,
            geojson_filepath=geojson_filepath,
            crop_type_key="Crop",
            class2idx=CLASS2IDX,
        )

    np.testing.assert_array_equal(input_mask, output_mask)


if __name__ == "__main__":
    test_load_geojson_mask()
