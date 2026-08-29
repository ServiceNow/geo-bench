# GEO-Bench: Toward Foundation Models for Earth Monitoring

GEO-Bench is a [ServiceNow Research](https://www.servicenow.com/research) project. 

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Language: Python](https://img.shields.io/badge/language-Python%203.9%2B-green?logo=python&logoColor=green)](https://www.python.org)

> **Use [GEO-Bench-2](https://github.com/The-AI-Alliance/GEO-Bench-2) for new work.** GEO-Bench-2 is
> the successor to this benchmark and where development continues. It is led by IBM and ServiceNow,
> covers more task types (classification, segmentation, regression, object detection, instance
> segmentation), and corrects known data issues in this version (see [Known issues](#known-issues)).
> See the [paper](https://arxiv.org/abs/2511.15658) and the
> [announcement](https://thealliance.ai/blog/geo-bench-2-from-performance-to-capability-rethinking-evaluation-in-geospatial-ai).
> This repository (GEO-Bench-1) stays available for reproducing the original
> [2023 paper](https://arxiv.org/abs/2306.03831).

GEO-Bench is a **G**eneral **E**arth **O**bservation benchmark for evaluating the performances of large pre-trained models on geospatial data. Read the [full paper](https://arxiv.org/abs/2306.03831) for usage details and evaluation of existing pre-trained vision models.

<img src="https://github.com/ServiceNow/geo-bench/raw/main/banner.png" width="500" />

## Installation

The version on PyPI (`1.0.0`) pins outdated upper bounds on its dependencies and predates the fixes
listed under [Known issues](#known-issues). Install from `main` instead:

```console
pip install git+https://github.com/ServiceNow/geo-bench.git
```

The released version is still available with `pip install geobench`.

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

Two datasets in `classification_v1.0` were published with incorrect band metadata, reported by
@gabrieltseng in [#28](https://github.com/ServiceNow/geo-bench/issues/28) and
[#29](https://github.com/ServiceNow/geo-bench/issues/29). The pixel values of each channel are
correct; the band name and wavelength attached to each channel are not, so code that selects
channels by band name rather than by index reads the wrong channel.

**`m-eurosat`** has 13 Sentinel-2 channels. The GeoTIFFs store `B8A` last, but the metadata assumes
it sits between `B08` and `B09`:

| Channel | Correct band | Published metadata |
| --- | --- | --- |
| 0–7 | B01–B08 | B01–B08 |
| 8 | B09 | B8A |
| 9 | B10 | B09 |
| 10 | B11 | B10 |
| 11 | B12 | B11 |
| 12 | B8A | B12 |

**`m-brick-kiln`** has 13 channels in the order exported by the
[source pipeline](https://github.com/mliu356/kiln-scaling): `B01, B02, B03, B04, B05, B07, B8A, B08,
B11, B12, TCI_R, TCI_G, TCI_B`. `B06`, `B09` and `B10` are absent, and the last three channels are
8-bit true-colour composites rather than reflectance bands. The metadata instead labels all 13 as
the canonical Sentinel-2 bands:

| Channel | Correct band | Published metadata |
| --- | --- | --- |
| 0–4 | B01–B05 | B01–B05 |
| 5 | B07 | B06 |
| 6 | B8A | B07 |
| 7 | B08 | B08 |
| 8 | B11 | B8A |
| 9 | B12 | B09 |
| 10 | TCI_R | B10 |
| 11 | TCI_G | B11 |
| 12 | TCI_B | B12 |

The converters under
[`make_benchmark/dataset_converters/`](https://github.com/ServiceNow/geo-bench/tree/main/make_benchmark/dataset_converters)
on `main` now write the correct metadata, but the data hosted on Hugging Face is unchanged.
GEO-Bench-2 uses corrected band metadata.

## Fine-tuning and reproducing experiments

See the code for reproducing experiments as a starting point for fine-tuning:

[geo-bench-experiments](https://github.com/ServiceNow/geo-bench-experiments)

## Visualizing Results

See the notebook [`baseline_results.ipynb`](https://github.com/ServiceNow/geo-bench/blob/main/geobench/baseline_results.ipynb) for an example of how to visualize the results.


