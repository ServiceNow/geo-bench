"""This script gives an example usage of the geobench package.
"""


import geobench

print("geobench dir:", geobench.GEO_BENCH_DIR)

for benchmark_name in ("classification_v0.9.1", "segmentation_v0.9.1"):
    print(f"Benchmark {benchmark_name}:\n")
    for task in geobench.task_iterator(benchmark_name=benchmark_name):
        print(f"Task {task.dataset_name}:\n  {task}\n")

        dataset = task.get_dataset(split="train")
        sample = dataset[0]

        print(f"Sample 0 named: {sample.sample_name}")
        for band in sample.bands:
            print(f"  {band.band_info.name}: {band.data.shape}")

        print("========================================\n")
