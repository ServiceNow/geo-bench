# How to Convert Datasets

This folder contains tools for converting heterogeneous datasets into a common format, which can be later read by the [GeobenchDataset](https://github.com/ServiceNow/geo-bench/blob/main/geobench/io/dataset.py) class.


## Adding New Datasets

1. Add a script for converting the dataset into `geobench/dataset_converters`. Take a look at `geobench/dataset_converters/brick_kiln.py` for a simple classification task example, or `geobench/dataset_converters/cv4a_kenya_crop_type.py` for a semantic segmentation example using multiple time-steps.

2. Add a jupyter notebook for visualizing contents of the datasets and printing some statistics, such as number of bands, timestamps, range of values for each band, and so on. Take a look at `geobench/dataset_converters/brick_kiln_inspect.ipynb` for a starting point.

## Downloading and Processing Existing Datasets

### Benin Smallholder Cashews
From top-level directory, run the following
```bash
python -m geobench.dataset_converters.benin_smallholder_cashews
```
Open the Jupyter notebook `geobench/dataset_converters/benin_smallholder_cashews_inspect.ipynb` to display stats and preview the dataset.


### Brick-Kiln
From top-level directory, run the following
```bash
python -m geobench.dataset_converters.brick_kiln
```
Open the Jupyter notebook `geobench/dataset_converters/brick_kiln_inspect.ipynb` to display stats and preview the dataset.


### Eurosat
From top-level directory, run the following
```bash
python -m geobench.dataset_converters.eurosat
```
Open the Jupyter notebook `geobench/dataset_converters/eurosat_inspect.ipynb` to display stats and preview the dataset.


### Kenya Crop
From top-level directory, run the following
```bash
python -m geobench.dataset_converters.cv4a_kenya_crop_type
```
Open the Jupyter notebook `geobench/dataset_converters/cv4a_kenya_crop_type_inspect.ipynb` to display stats and preview the dataset.


## Computing Data Statistics

Coming soon – need to compute mean, std, percentiles about datasets and store them in task_specs.pkl
