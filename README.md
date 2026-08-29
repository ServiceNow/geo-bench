# GEO-Bench: Toward Foundation Models for Earth Monitoring

GEO-Bench is a [ServiceNow Research](https://www.servicenow.com/research) project. 

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Language: Python](https://img.shields.io/badge/language-Python%203.9%2B-green?logo=python&logoColor=green)](https://www.python.org)

GEO-Bench is a **G**eneral **E**arth **O**bservation benchmark for evaluating the performances of large pre-trained models on geospatial data. Read the [full paper](https://arxiv.org/abs/2306.03831) for usage details and evaluation of existing pre-trained vision models.

<img src="https://github.com/ServiceNow/geo-bench/raw/main/banner.png" width="500" />

## Installation

The release on PyPI (`1.0.0`) pins outdated upper bounds on its dependencies, which conflicts with
current versions of the scientific-Python stack ([#20](https://github.com/ServiceNow/geo-bench/pull/20)
relaxes them on `main`). Install from `main`:

```console
pip install git+https://github.com/ServiceNow/geo-bench.git
```

The PyPI release is still available with `pip install geobench`.

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

## Known issues

The `m-eurosat` and `m-brick-kiln` datasets in `classification_v1.0` record the wrong Sentinel-2
band for most of their channels. The pixel data is unaffected; only the band name and wavelength
stored for each channel are wrong, so selecting channels by band name returns the wrong channel.
Reported by @gabrieltseng in [#28](https://github.com/ServiceNow/geo-bench/issues/28) and
[#29](https://github.com/ServiceNow/geo-bench/issues/29).

The 13 channels are actually in this order:

| Channel | `m-eurosat` | `m-brick-kiln` |
| --- | --- | --- |
| 0–4 | B01–B05 | B01–B05 |
| 5 | B06 | B07 |
| 6 | B07 | B8A |
| 7 | B08 | B08 |
| 8 | B09 | B11 |
| 9 | B10 | B12 |
| 10 | B11 | TCI_R |
| 11 | B12 | TCI_G |
| 12 | B8A | TCI_B |

The stored metadata instead labels all 13 channels, in both datasets, as
`B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12`. In `m-brick-kiln` the source
pipeline ([mliu356/kiln-scaling](https://github.com/mliu356/kiln-scaling)) does not export `B06`,
`B09` or `B10`, and its last three channels are 8-bit true-colour composites rather than reflectance
bands.

The converters in `make_benchmark/dataset_converters/` now record the order above; the data on
Hugging Face is not regenerated, so apply this mapping when loading it. Loading either dataset
through `GeobenchDataset` emits a warning to this effect.

## Fine-tuning and reproducing experiments

See the code for reproducing experiments as a starting point for fine-tuning:

[geo-bench-experiments](https://github.com/ServiceNow/geo-bench-experiments)

## Visualizing Results

See the notebook [`baseline_results.ipynb`](https://github.com/ServiceNow/geo-bench/blob/main/geobench/baseline_results.ipynb) for an example of how to visualize the results.


