# How to Convert Datasets

## Folder Configuration
Optionally, start by setting folders for *downloading* and *converting* datasets
```
export CC_BENCHMARK_SOURCE_DATASETS=/tmp/dataset
export CC_BENCHMARK_CONVERTED_DATASETS=/tmp/converted_dataset
```
Otherwise, they default to `~/dataset` and `~/converted_dataset`.

## Adding New Datasets

Take a look at `ccb/dataset_converters/brick_kiln.py` for a simple classification task example, 
or `ccb/dataset_converters/cv4a_kenya_crop_type.py` for a semantic segmentation example using multiple time-steps .

## Downloading and Processing Existing Datasets

### Brick-Kiln
From top-level directory, run the following
```bash
python -m ccb.dataset_converters.brick_kiln
```

Open the Jupyter notebook `ccb/dataset_converters/brick_kiln_inspect.ipynb` to display stats and preview the dataset.
