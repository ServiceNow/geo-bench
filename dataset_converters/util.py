import os

# TODO replace by environment variable CC_BENCHMARK_SOURCE_DATASETS
src_datasets_dir = os.environ.get('CC_BENCHMARK_SOURCE_DATASETS', os.path.expanduser("~/dataset/"))
dst_datasets_dir = os.environ.get('CC_BENCHMARK_CONVERTED_DATASETS', os.path.expanduser("~/converted_dataset/"))
