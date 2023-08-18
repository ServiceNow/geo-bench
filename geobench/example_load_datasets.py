"""This script gives an example usage of the geobench package.
"""

from geobench import io

for task in io.task_iterator(benchmark_name="classification_v0.8.2"):
    print(f"Task {task.dataset_name}:\n  {task}\n")

    dataset = task.get_dataset(split="train")
    sample = dataset[0]

    print(f"Sample 0 named: {sample.sample_name}")
    for band in sample.bands:
        print(f"  {band.band_info.name}: {band.data.shape}")

    print("========================================\n")
