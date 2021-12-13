import os

# TODO replace by environment variable CC_BENCHMARK_SOURCE_DATASETS
src_datasets_dir = os.path.expanduser("~/dataset/")
dst_datasets_dir = os.path.expanduser("~/converted_dataset/")

SENTINEL2_BAND_NAMES = """\
01 - Coastal aerosol
02 - Blue
03 - Green
04 - Red
05 - Vegetation Red Edge
06 - Vegetation Red Edge
07 - Vegetation Red Edge
08 - NIR
08A - Vegetation Red Edge
09 - Water vapour
10 - SWIR - Cirrus
11 - SWIR
12 - SWIR
""".split("\n")

SENTINEL2_CENTRAL_WAVELENGTHS = """\
0.443
0.49
0.56
0.665
0.705
0.74
0.783
0.842
0.865
0.945
1.375
1.61
2.19""".split("\n")

SENTINEL2_CENTRAL_WAVELENGTHS = [float(wl) for wl in SENTINEL2_CENTRAL_WAVELENGTHS]
