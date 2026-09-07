
## Set up the environment

The converters need more than the geobench package itself: torchgeo and torchvision to fetch
several of the source datasets, Pillow, tifffile, pyproj and xmltodict for the others, and
ipyleaflet and ipyplot for the inspection notebooks. They are declared as the `make-benchmark`
dependency group, so from the repository root run

```shell
uv sync --group make-benchmark
```

The group tracks current releases of those packages. The converters were written against
torchgeo 0.5 and torch 1.12, so a converter may need adjusting to a changed upstream API
before it runs.

## Download original datasets
Download each original dataset and store them in GEO_BENCH_DIR/source.

Follow the procedure described at the beginning of the converter on how to download and extract. 
TorchGeo datasets should download automatically.

## convert the datasets
Use the converter to conver the original datasets to the format used by the benchmark.

## label_map and label_stats

run label_map.py to create the required label statistics. These are use for resampling

run
```shell
python make_benchmark/label_map.py
```

## create the benchmark
Needs to have the datasets downloaded and converted already

run 

```shell
python make_benchmark/create_benchmark.py
```