# Downloaded from "https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg13/brick_kiln.html"

import h5py
from pathlib import Path
import rasterio
import util  # TODO reorganize packages and use absolute import instead of relative


src_dataset_dir = Path(util.src_datasets_dir, "brick_kiln_v1.0")
dst_dataset_dir = Path(util.dst_datasets_dir, "brick_kiln_v1.0")
dst_dataset_dir.mkdir(exist_ok=True, parents=True)


def load_examples_bloc(file_path):
    """Load a .h5py bloc of images with their labels."""
    file_id = file_path.stem.split('_')[1]

    with h5py.File(file_path) as data:

        images = data['images'][:]
        labels = data['labels'][:]
        bounds = data['bounds'][:]

        return images, labels, bounds, file_id


file_list = list(src_dataset_dir.iterdir())
for i, file_path in enumerate(file_list):
    if file_path.suffix != '.hdf5':
        continue

    images, labels, bounds, file_id = load_examples_bloc(file_path)

    print("%d/%d %s. Shape = %s." % (i + 1, len(file_list), file_path.name, str(images.shape)))

    if images.shape[0] == 0:
        print("Skipping block of shape 0. Shape = %s" % (str(images.shape)))
        continue

    for i, (image, label, box) in enumerate(zip(images, labels, bounds)):
        img_path = Path(dst_dataset_dir, 'examples_%s_%d.tif' % (file_id, i))

        image = util.swap_band_axes_to_last(image)

        # box = lon_top_left,lat_top_left,lon_bottom_right,lat_bottom_right
        # from_bounds = west, south, east, north
        transform = rasterio.transform.from_bounds(
            box[0], box[3], box[2], box[1], image.shape[1], image.shape[2])
        sample = util.Sample(
            image, int(label), 10, util.SENTINEL2_BAND_NAMES, util.SENTINEL2_CENTRAL_WAVELENGTHS, transform)

        sample.to_geotiff(img_path)


# def read_list_eval_partition(csv_file):
#     with open(csv_file) as fd:
#         reader = csv.reader(fd, delimiter=',')
#         data = []
#         for i, row in enumerate(reader):
#             if i == 0:
#                 continue
#             data.append([float(e) for e in row])

#     data = np.array(data)

#     y = data[:, 0].astype(np.int)
#     partition = data[:, 1].astype(np.int)
#     hdf5_file = data[:, 2].astype(np.int)
#     hdf5_idx = data[:, 3].astype(np.int)
#     gps = data[:, 4:8]
#     indices = data[:, 8:11].astype(np.int)

#     id_map = {}
#     for i, (file_id, sample_idx) in enumerate(zip(hdf5_file, hdf5_idx)):
#         id_map[(file_id, sample_idx)] = i

#     return y, partition, hdf5_file, hdf5_idx, gps, indices, id_map


# y, partition, hdf5_file, hdf5_idx, gps, indices, id_map = read_list_eval_partition(
#     Path(src_dataset_dir, 'list_eval_partition.csv'))
