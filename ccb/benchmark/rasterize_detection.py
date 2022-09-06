"""Rasterize detection."""
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


def rasterize_box(boxes: List[Dict[str, int]], img_shape: Tuple[int, int], scale=1) -> "np.typing.NDArray[np.int_]":
    """Rasterize box.

    Args:
        boxes:
        img_shape:
        scale:

    Returns:
        rasterized boxes
    """
    im = Image.new(mode="L", size=img_shape)

    ctxt = ImageDraw.Draw(im)
    for obj in boxes:
        if isinstance(obj, dict) and "xmin" in obj:

            if scale != 1:
                range = np.array([obj["xmax"] - obj["xmin"], obj["ymax"] - obj["ymin"]])
                d_x, d_y = range * (1 - scale) / 2.0
            else:
                d_x, d_y = (0, 0)

            coord = np.array([[obj["xmin"] + d_x, obj["ymin"] + d_y], [obj["xmax"] - d_x, obj["ymax"] - d_y]])
            ctxt.ellipse(list(coord.flat), fill=1)

    return np.array(im)


def point_to_boxes(points, radius):
    """Convert point to boxes.

    Args:
        points:
        radius:

    Returns:
        bounding boxes
    """
    boxes = []
    for point in points:
        boxes.append(
            {"xmin": point[0] - radius, "ymin": point[1] - radius, "xmax": point[0] + radius, "ymax": point[1] + radius}
        )
    return boxes


if __name__ == "__main__":

    boxes = [{"xmin": 2, "ymin": 3, "xmax": 6, "ymax": 12}]
    raster = rasterize_box(boxes, img_shape=(20, 20))
    print(raster.dtype)
    print(raster)
