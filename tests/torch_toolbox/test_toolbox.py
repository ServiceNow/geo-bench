import os
import pickle
import tempfile
from pathlib import Path

import pytest

from ccb import io
from ccb.experiment.experiment import Job
from ccb.io import mnist_task_specs
from ccb.torch_toolbox import trainer
from ccb.torch_toolbox.model_generators import conv4, py_segmentation_generator, timm_generator


def train_job_on_task(
    model_generator, task_specs, threshold, check_logs=True, logger=None, metric_name="Accuracy", **kwargs
):
    """Based on a job train model_generator on task.

    Args:
        model_generator: model_generator that has been instantiated and called with desired hparams
        task_specs: task specifications which to train model on

    """
    with tempfile.TemporaryDirectory(prefix="test") as job_dir:
        job = Job(job_dir)
        task_specs.save(job.dir)

        hparams = model_generator.base_hparams
        job.save_hparams(hparams)

        trainer.train(model_gen=model_generator, job_dir=job_dir)

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


@pytest.mark.slow
def test_toolbox_seeds():
    with open(
        os.path.join("tests", "data", "ccb-test-classification", "brick_kiln_v1.0", "task_specs.pkl"), "rb"
    ) as fd:
        task_specs = pickle.load(fd)
    hparams = {
        "backbone": "resnet18",
        "pretrained": True,
        "logger": "csv",
        "lr_backbone": 1e-3,
        "lr_head": 1e-2,
        "optimizer": "sgd",
        "momentum": 0.9,
        "batch_size": 32,
        "max_epochs": 1,
        "num_workers": 0,
        "seed": 1,
    }

    metrics1 = train_job_on_task(conv4.model_generator(hparams), task_specs, 0.05, deterministic=True)
    metrics2 = train_job_on_task(conv4.model_generator(hparams), task_specs, 0.05, deterministic=True)

    assert metrics1["test_Accuracy"] == metrics2["test_Accuracy"]


@pytest.mark.optional
def test_toolbox_wandb():
    hparams = {
        "backbone": "resnet18",
        "pretrained": True,
        "logger": "csv",
        "lr_backbone": 1e-3,
        "lr_head": 1e-2,
        "optimizer": "sgd",
        "momentum": 0.9,
        "batch_size": 32,
        "max_epochs": 1,
        "num_workers": 0,
        "seed": 1,
    }
    train_job_on_task(conv4.model_generator(hparams), mnist_task_specs, 0.05, logger="wandb")


@pytest.mark.slow
def test_toolbox_brick_kiln():
    with open(
        os.path.join("tests", "data", "ccb-test-classification", "brick_kiln_v1.0", "task_specs.pkl"), "rb"
    ) as fd:
        task_specs = pickle.load(fd)
    hparams = {
        "backbone": "resnet18",
        "pretrained": True,
        "lr_backbone": 1e-3,
        "lr_head": 1e-2,
        "logger": "csv",
        "optimizer": "sgd",
        "momentum": 0.9,
        "batch_size": 32,
        "max_epochs": 1,
        "band_names": ["red", "green", "blue"],
        "format": "hdf5",
        "num_workers": 0,
        "seed": 1,
    }
    train_job_on_task(conv4.model_generator(hparams), task_specs, 0.40)


def test_toolbox_segmentation():
    with open(
        os.path.join("tests", "data", "ccb-test-segmentation", "cvpr_chesapeake_landcover", "task_specs.pkl"),
        "rb",
    ) as fd:
        task_specs = pickle.load(fd)

    hparams = {
        "input_size": (3, 64, 64),  # FIXME
        "pretrained": True,
        "lr_backbone": 1e-3,
        "lr_head": 1e-2,
        "logger": "csv",
        "optimizer": "sgd",
        "head_type": "linear",
        "loss_type": "crossentropy",
        "batch_size": 8,
        "max_epochs": 1,
        "encoder_type": "resnet18",
        "decoder_type": "Unet",
        "decoder_weights": "imagenet",
        "format": "hdf5",
        "num_workers": 0,
        "seed": 1,
    }

    train_job_on_task(py_segmentation_generator.model_generator(hparams), task_specs, 0.50, check_logs=False)


# this test is too slow
@pytest.mark.slow
def test_toolbox_timm():
    hparams = {
        "backbone": "resnet18",
        "pretrained": True,
        "logger": "csv",
        "lr_backbone": 1e-3,
        "lr_head": 1e-2,
        "optimizer": "sgd",
        "momentum": 0.9,
        "batch_size": 32,
        "max_epochs": 1,
        "band_names": ["red", "green", "blue"],
        "num_workers": 0,
        "seed": 1,
    }
    with open(
        os.path.join("tests", "data", "ccb-test-classification", "brick_kiln_v1.0", "task_specs.pkl"), "rb"
    ) as fd:
        task_specs = pickle.load(fd)
    train_job_on_task(timm_generator.model_generator(hparams), task_specs, 0.40)


def test_toolbox_getitem():
    test_dirs = ["ccb-test-classification", "ccb-test-segmentation"]
    for benchmark_name in test_dirs:
        for task in io.task_iterator(benchmark_name):
            dataset = task.get_dataset(split="valid")
            data = dataset[0]
            if benchmark_name not in test_dirs:
                assert isinstance(data, dict)
            else:
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
