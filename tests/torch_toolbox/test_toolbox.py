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

        if check_logs:

            metrics = job.get_metrics()
            print(metrics)
            assert (
                float(metrics[f"test_{metric_name}"]) > threshold
            )  # has to be better than random after seeing 20 batches
            return metrics

        return None


@pytest.mark.slow
def test_toolbox_mnist():
    hparams = {
        "backbone": "resnet18",
        "pretrained": True,
        "logger": "csv",
        "lr_backbone": 1e-6,
        "lr_head": 1e-4,
        "optimizer": "sgd",
        "momentum": 0.9,
        "batch_size": 32,
        "max_epochs": 1,
        "num_workers": 0,
        "gpu": None,
    }
    train_job_on_task(conv4.model_generator(hparams), mnist_task_specs, 0.05)


@pytest.mark.slow
def test_toolbox_seeds():
    hparams = {
        "backbone": "resnet18",
        "pretrained": True,
        "logger": "csv",
        "lr_backbone": 1e-6,
        "lr_head": 1e-4,
        "optimizer": "sgd",
        "momentum": 0.9,
        "batch_size": 32,
        "max_epochs": 1,
        "num_workers": 0,
        "seed": 1,
        "gpu": None,
    }
    metrics1 = train_job_on_task(conv4.model_generator(hparams), mnist_task_specs, 0.05, deterministic=True)
    metrics2 = train_job_on_task(conv4.model_generator(hparams), mnist_task_specs, 0.05, deterministic=True)
    hparams.update({"seed": 2})
    metrics3 = train_job_on_task(conv4.model_generator(hparams), mnist_task_specs, 0.05, deterministic=True)
    assert metrics1["test_Accuracy"] == metrics2["test_Accuracy"] != metrics3["test_Accuracy"]


@pytest.mark.optional
def test_toolbox_wandb():
    hparams = {
        "backbone": "resnet18",
        "pretrained": True,
        "logger": "csv",
        "lr_backbone": 1e-6,
        "lr_head": 1e-4,
        "optimizer": "sgd",
        "momentum": 0.9,
        "batch_size": 32,
        "max_epochs": 1,
        "num_workers": 0,
        "gpu": None,
    }
    train_job_on_task(conv4.model_generator(hparams), mnist_task_specs, 0.05, logger="wandb")


@pytest.mark.slow
def test_toolbox_brick_kiln():
    with open(os.path.join("tests", "data", "brick_kiln_v1.0", "task_specs.pkl"), "rb") as fd:
        task_specs = pickle.load(fd)
    hparams = {
        "backbone": "resnet18",
        "pretrained": True,
        "lr_backbone": 1e-6,
        "logger": "csv",
        "lr_head": 1e-4,
        "optimizer": "sgd",
        "momentum": 0.9,
        "batch_size": 32,
        "max_epochs": 1,
        "band_names": ["red", "green", "blue"],
        "format": "hdf5",
        "num_workers": 0,
        "gpu": None,
    }
    train_job_on_task(conv4.model_generator(hparams), task_specs, 0.40)


# def test_toolbox_segmentation():
#     with open(os.path.join("tests", "data", "cvpr_chesapeake_landcover", "task_specs.pkl"), "rb") as fd:
#         task_specs = pickle.load(fd)

#     hparams = {
#         "input_size": (3, 64, 64),  # FIXME
#         "pretrained": True,
#         "lr_backbone": 1e-5,
#         "lr_head": 1e-4,
#         "logger": "csv",
#         "optimizer": "sgd",
#         "head_type": "linear",
#         "loss_type": "crossentropy",
#         "batch_size": 8,
#         "max_epochs": 1,
#         "encoder_type": "resnet18",
#         "decoder_type": "Unet",
#         "decoder_weights": "imagenet",
#         "format": "hdf5",
#         "num_workers": 0,
#         "gpu": None,
#     }

#     train_job_on_task(py_segmentation_generator.model_generator(hparams), task_specs, 0.50, check_logs=False)


# # this test is too slow
# @pytest.mark.slow
# def test_toolbox_timm():
#     hparams = {
#         "backbone": "resnet18",
#         "pretrained": True,
#         "logger": "csv",
#         "lr_backbone": 1e-6,
#         "lr_head": 1e-4,
#         "optimizer": "sgd",
#         "momentum": 0.9,
#         "batch_size": 32,
#         "max_epochs": 1,
#         "band_names": ["red", "green", "blue"],
#         "num_workers": 0,
#         "gpu": None,
#     }
#     with open(os.path.join("tests", "data", "brick_kiln_v1.0", "task_specs.pkl"), "rb") as fd:
#         task_specs = pickle.load(fd)
#     train_job_on_task(timm_generator.model_generator(hparams), task_specs, 0.70)


def test_toolbox_bigearthnet():
    hparams = {
        "backbone": "resnet18",
        "pretrained": False,
        "logger": "csv",
        "lr_backbone": 1e-1,
        "lr_head": 1e-1,
        "nesterov": True,
        "optimizer": "sgd",
        "momentum": 0.9,
        "batch_size": 32,
        "max_epochs": 1,
        "fast_dev_run": True,
        "logger": "csv",
        "format": "hdf5",
        "band_names": ["red", "green", "blue"],
        "format": "hdf5",
        "num_workers": 0,
        "gpu": None,
    }
    with open(os.path.join("tests", "data", "brick_kiln_v1.0", "task_specs.pkl"), "rb") as fd:
        task_specs = pickle.load(fd)
    train_job_on_task(timm_generator.model_generator(hparams), task_specs, 0.70, check_logs=False)


def test_toolbox_getitem():
    benchmark_dir = Path(os.path.join("tests", "data"))
    for benchmark_name in ("test", "imagenet", "ccb-test"):
        for task in io.task_iterator(benchmark_name, benchmark_dir):
            dataset = task.get_dataset(split="valid")
            data = dataset[0]
            if benchmark_name != "ccb-test":
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
