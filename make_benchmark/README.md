
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