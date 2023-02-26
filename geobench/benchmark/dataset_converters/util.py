"""Utility functions for dataset converters."""

import pyproj
import rasterio


def center_to_transform(lat_center, lon_center, radius_in_meter, img_shape):
    """Convert center point and radius to rasterio transform, assuming lat long coordinates."""
    geod = pyproj.Geod(ellps="clrk66")
    lon, lat, baz = geod.fwd([lon_center] * 4, [lat_center] * 4, [0, 90, 180, 270], [radius_in_meter] * 4)
    north, east, south, west = lat[0], lon[1], lat[2], lon[3]
    transform = rasterio.transform.from_bounds(west, south, east, north, *img_shape)
    return transform
