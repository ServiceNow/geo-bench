# Climate change benchmark

GeoBench is a ServiceNow Research project.
 
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Language: Python](https://img.shields.io/badge/language-Python%203.7%2B-green?logo=python&logoColor=green)](https://www.python.org)

<img src="https://github.com/ElementAI/climate-change-benchmark/raw/main/banner.png" />

A set of climate change tasks to benchmark self-supervised learning algorithms.


## Visualizing the Content of the Benchmark

Simply run the notebook `instpect_benchmark.ipynb`

## Basic Usage

```bash
export CCB_DIR=/path/to/the/benchmark_dir
```

```python
from ccb import io

def my_transform(sample):
    data, _ = sample.pack_to_3d(band_names=("red", "green", "blue"))
    return data

for task in io.task_iterator(io.CCB_DIR / "classification_v0.7"):
    print(task)
    dataset = task.get_dataset(split="train", partition_name="default", transform=my_transform)
    data = dataset[0] # load and transform the first sample
    print(data.shape)
```

## Contributing

We welcome your contributions! Please see [our wiki](https://github.com/ElementAI/climate-change-benchmark/wiki#instructions-for-contributing) for instructions on how to contribute.

## Getting Started

Quick jump to [launching experiments](https://github.com/ElementAI/climate-change-benchmark/wiki/Running-Experiments-on-EAI-Toolkit).

### Downloading and Converting Datasets

Read these [instructions](https://github.com/ElementAI/climate-change-benchmark/tree/main/ccb/dataset_converters#readme) for adding a new dataset or downloading & processing an existing dataset.
