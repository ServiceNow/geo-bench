# GEO-Bench: Toward Foundation Models for Earth Monitoring

GEO-Bench is a [ServiceNow Research](https://www.servicenow.com/research) project. 

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Language: Python](https://img.shields.io/badge/language-Python%203.9%2B-green?logo=python&logoColor=green)](https://www.python.org)

GEO-Bench is a **G**eneral **E**arth **O**bservation benchmark for evaluating the performances of large pre-trained models on geospatial data. Read the [full paper](https://arxiv.org/abs/2306.03831) for usage details and evaluation of existing pre-trained vision models.

<img src="https://github.com/ServiceNow/geo-bench/raw/main/banner.png" width="500" />

## Installation

You can install GEO-Bench with [pip](https://pip.pypa.io/):

```console
pip install geobench
```

Note: Python 3.9+ is required.

## Downloading the data

Set `$GEO_BENCH_DIR` to your preferred location. If not set, it will be stored in `$HOME/dataset/geobench`.

Next, use the [download script](https://github.com/ServiceNow/geo-bench/blob/main/geobench/geobench_download.py). This will automatically download from [Hugging Face](https://huggingface.co/datasets/recursix/geo-bench-1.0)

Run the command:

```console
geobench-download
```

You need ~65 GB of free disk space for download and unzip (once all .zip are deleted it takes 57GB).
If some files are already downloaded, it will verify the md5 checksum. Feel free to restart the downloader if it is interrupted.

## Test installation
You can run tests. 
Note: Make sure the benchmark is downloaded before launching tests.

```console
pip install pytest
```

```console
geobench-test
```

## Loading Datasets

See [`example_load_dataset.py`](https://github.com/ServiceNow/geo-bench/blob/main/geobench/example_load_datasets.py) for how to iterate over datasets.

```python
import geobench

for task in geobench.task_iterator(benchmark_name="classification_v1.0"):
    dataset = task.get_dataset(split="train")
    sample = dataset[0]
    for band in sample.bands:
        print(f"{band.band_info.name}: {band.data.shape}")
```

## Fine-tuning and reproducing experiments

See the code for reproducing experiments as a starting point for fine-tuning:

[geo-bench-experiments](https://github.com/ServiceNow/geo-bench-experiments)

## Visualizing Results

See the notebook [`baseline_results.ipynb`](https://github.com/ServiceNow/geo-bench/blob/main/geobench/baseline_results.ipynb) for an example of how to visualize the results.


