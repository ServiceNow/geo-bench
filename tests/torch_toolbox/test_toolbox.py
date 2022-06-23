import os
import pickle
import tempfile
from datetime import datetime


import pytest

from ruamel.yaml import YAML
from ccb import io
from ccb.experiment.experiment import Job
from ccb.torch_toolbox import trainer


def train_job_on_task(config, hparams, task_specs, threshold, check_logs=True, metric_name="Accuracy", **kwargs):
    """Based on a job train model_generator on task.

    Args:
        model_generator: model_generator that has been instantiated and called with desired hparams
        config: config file
        hparams: hparam file
        task_specs: task specifications which to train model on

    """
    with tempfile.TemporaryDirectory(prefix="test") as job_dir:
        # job_dir = f"{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"
        # os.makedirs(job_dir, exist_ok=True)

        job = Job(job_dir)
        task_specs.save(job.dir)

        job.save_hparams(hparams)
        job.save_config(config)

        trainer.train(job_dir=job_dir)

        print(task_specs.benchmark_name)
        if check_logs:

            metrics = job.get_metrics()
            print(metrics)
            print(task_specs.benchmark_name)
            print(hparams)
            assert (
                float(metrics[f"test_{metric_name}"]) > threshold
            )  # has to be better than random after seeing 20 batches
            return metrics

        return None


def test_toolbox_segmentation():
    with open(
        os.path.join("tests", "data", "ccb-test-segmentation", "cvpr_chesapeake_landcover", "task_specs.pkl"), "rb"
    ) as fd:
        task_specs = pickle.load(fd)

    yaml = YAML()
    with open(os.path.join("tests", "configs", "base_segmentation.yaml"), "r") as yamlfile:
        config = yaml.load(yamlfile)

    with open(os.path.join("tests", "configs", "segmentation_hparams.yaml"), "r") as yamlfile:
        hparams = yaml.load(yamlfile)

    train_job_on_task(config=config, hparams=hparams, task_specs=task_specs, threshold=0.05, metric_name="JaccardIndex")


# this test is too slow
@pytest.mark.slow
def test_toolbox_timm():
    with open(
        os.path.join("tests", "data", "ccb-test-classification", "brick_kiln_v1.0", "task_specs.pkl"), "rb"
    ) as fd:
        task_specs = pickle.load(fd)

    yaml = YAML()
    with open(os.path.join("tests", "configs", "base_classification.yaml"), "r") as yamlfile:
        config = yaml.load(yamlfile)

    with open(os.path.join("tests", "configs", "classification_hparams.yaml"), "r") as yamlfile:
        hparams = yaml.load(yamlfile)

    train_job_on_task(config=config, hparams=hparams, task_specs=task_specs, threshold=0.40)


def test_toolbox_getitem():
    benchmarks = ["ccb-test-classification", "ccb-test-segmentation"]
    test_dirs = [os.path.join("tests", "data", benchmark) for benchmark in benchmarks]
    for benchmark_dir in test_dirs:
        for task in io.task_iterator(benchmark_dir):
            dataset = task.get_dataset(split="valid", benchmark_dir=benchmark_dir)
            data = dataset[0]
            assert isinstance(data, io.Sample)


if __name__ == "__main__":
    # test_toolbox_timm()
    # test_toolbox_brick_kiln()
    # test_toolbox_wandb()
    # test_toolbox_mnist()
    # test_toolbox_getitem()
    # test_toolbox_seeds()
    # test_toolbox_segmentation()
    # test_toolbox_bigearthnet()
    pass
