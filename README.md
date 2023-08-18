# GEO-Bench: Toward Foundation Models for Earth Monitoring

GeoBench is a [ServiceNow Research](https://www.servicenow.com/research) project. 

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Language: Python](https://img.shields.io/badge/language-Python%203.7%2B-green?logo=python&logoColor=green)](https://www.python.org)

GEO-Bench is a General Earth Observation benchmark for evaluating the performances of large pre-trained models on geospatial data. Read the [full paper](https://arxiv.org/abs/2306.03831) for usage details and evaluation of existing pre-trained vision models.

<img src="https://github.com/ServiceNow/geo-bench/raw/main/banner.png" width="500" />

## Installation

You can install GEO-Bench with [pip](https://pip.pypa.io/):

```console
$ pip install geo-benchmark
```

## Downloading the data

Set `$GEO_BENCH_DIR` to your preferred location. If not set, it will be stored in `$HOME/dataset/geobench`.

Next, use the [download script](https://github.com/ServiceNow/geo-bench/blob/main/geobench/download_geobench.py). This will automatically download from [Zenodo](https://zenodo.org/communities/geo-bench/)

```console
cd geobench
python download_geobench.py
```

## Loading Datasets

See [`example_load_dataset.py`](https://github.com/ServiceNow/geo-bench/blob/main/geobench/example_load_datasets.py) for how to iterate over datasets.

```python
from geobench import io

for task in io.task_iterator(benchmark_name="classification_v0.9.0"):
    dataset = task.get_dataset(split="train")
    sample = dataset[0]
    for band in sample.bands:
        print(f"  {band.band_info.name}: {band.data.shape}")

```
## Visualizing Results

See the notebook [`baseline_results.ipynb`](https://github.com/ServiceNow/geo-bench/blob/main/geobench/baseline_results.ipynb) for an example of how to visualize the results.


