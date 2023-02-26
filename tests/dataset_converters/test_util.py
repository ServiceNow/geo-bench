from geobench.benchmark.dataset_converters import util
import numpy as np

def transform_to_center(transform, img_shape):
    lon_lat = transform * (np.array(img_shape) / 2.)
    return lon_lat[::-1]

def test_center_to_transform():
    point_lat_lon = 45.630001, -73.519997
    img_shape = 100, 100
    spatial_resolution = 10
    radius_in_meter = spatial_resolution * img_shape[0] / 2


    transfrorm = util.center_to_transform(*point_lat_lon, radius_in_meter, img_shape)

    point_lat_lon_ = transform_to_center(transfrorm, np.array(img_shape)*1.)

    assert np.allclose(point_lat_lon, point_lat_lon_)


if __name__ == '__main__':
    test_center_to_transform()