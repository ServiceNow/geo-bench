import pickle
import tempfile

import pytest
from ccb.experiment.experiment import Job
from ccb.io import mnist_task_specs
from ccb.torch_toolbox.model_generators import conv4_test, timm_generator, py_segmentation_generator
from ccb.torch_toolbox import trainer
from ccb import io
from pathlib import Path


def train_job_on_task(
    model_generator, task_specs, threshold, check_logs=True, logger=None, metric_name="Accuracy", **kwargs
):
    with tempfile.TemporaryDirectory(prefix="ccb_mnist_test") as job_dir:
        job = Job(job_dir)
        task_specs.save(job.dir)

        hparams = model_generator.hp_search(task_specs)[0][0]
        if logger is not None:
            hparams["logger"] = logger
        hparams.update(kwargs)
        job.save_hparams(hparams)

        trainer.train(model_gen=model_generator, job_dir=job_dir, wandb_mode="standard")
        hparams = job.hparams

        if check_logs:
            hparams = job.hparams

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
        "lr_backbone": 1e-6,
        "lr_head": 1e-4,
        "optimizer": "sgd",
        "momentum": 0.9,
        "batch_size": 32,
        "max_epochs": 1,
    }
    train_job_on_task(conv4_test.model_generator(hparams), mnist_task_specs, 0.05)


@pytest.mark.slow
def test_toolbox_seeds():
    hparams = {
        "backbone": "resnet18",
        "pretrained": True,
        "lr_backbone": 1e-6,
        "lr_head": 1e-4,
        "optimizer": "sgd",
        "momentum": 0.9,
        "batch_size": 32,
        "max_epochs": 1,
    }
    metrics1 = train_job_on_task(
        conv4_test.model_generator(hparams), mnist_task_specs, 0.05, deterministic=True, seed=1
    )
    metrics2 = train_job_on_task(
        conv4_test.model_generator(hparams), mnist_task_specs, 0.05, deterministic=True, seed=1
    )
    metrics3 = train_job_on_task(
        conv4_test.model_generator(hparams), mnist_task_specs, 0.05, deterministic=True, seed=2
    )
    assert metrics1["test_Accuracy"] == metrics2["test_Accuracy"] != metrics3["test_Accuracy"]


@pytest.mark.optional
def test_toolbox_wandb():
    hparams = {
        "backbone": "resnet18",
        "pretrained": True,
        "lr_backbone": 1e-6,
        "lr_head": 1e-4,
        "optimizer": "sgd",
        "momentum": 0.9,
        "batch_size": 32,
        "max_epochs": 1,
    }
    train_job_on_task(conv4_test.model_generator(hparams), mnist_task_specs, 0.05, logger="wandb")


@pytest.mark.slow
@pytest.mark.skipif(
    not Path(io.CCB_DIR / "ccb-test" / "brick_kiln_v1.0").exists(), reason="Requires presence of the benchmark."
)
def test_toolbox_brick_kiln():
    with open(Path(io.CCB_DIR) / "ccb-test" / "brick_kiln_v1.0" / "task_specs.pkl", "rb") as fd:
        task_specs = pickle.load(fd)
    hparams = {
        "backbone": "resnet18",
        "pretrained": True,
        "lr_backbone": 1e-6,
        "lr_head": 1e-4,
        "optimizer": "sgd",
        "momentum": 0.9,
        "batch_size": 32,
        "max_epochs": 1,
        "band_names": ["red", "green", "blue"],
    }
    train_job_on_task(conv4_test.model_generator(hparams), task_specs, 0.40)


def test_toolbox_segmentation():
    with open(Path(io.CCB_DIR) / "segmentation_v0.2/cvpr_chesapeake_landcover/task_specs.pkl", "rb") as fd:
        task_specs = pickle.load(fd)
    train_job_on_task(py_segmentation_generator.model_generator(), task_specs, 0.70, check_logs=False)


# this test is too slow


@pytest.mark.slow
@pytest.mark.skipif(
    not Path(io.CCB_DIR / "ccb-test" / "brick_kiln_v1.0").exists(), reason="Requires presence of the benchmark."
)
def test_toolbox_timm():
    hparams = {
        "backbone": "resnet18",
        "pretrained": True,
        "lr_backbone": 1e-6,
        "lr_head": 1e-4,
        "optimizer": "sgd",
        "momentum": 0.9,
        "batch_size": 32,
        "max_epochs": 1,
        "band_names": ["red", "green", "blue"],
    }
    with open(Path(io.CCB_DIR) / "ccb-test" / "brick_kiln_v1.0" / "task_specs.pkl", "rb") as fd:
        task_specs = pickle.load(fd)
    train_job_on_task(timm_generator.model_generator(hparams), task_specs, 0.70)


def test_toolbox_bigearthnet():
    hparams = {
        "backbone": "resnet18",
        "pretrained": False,
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
    }
    with open(Path(io.CCB_DIR) / "classification_v0.4" / "bigearthnet" / "task_specs.pkl", "rb") as fd:
        task_specs = pickle.load(fd)
    train_job_on_task(timm_generator.model_generator(hparams), task_specs, 0.70, check_logs=False)


@pytest.mark.skipif(
    not Path(io.CCB_DIR / "ccb-test" / "brick_kiln_v1.0").exists() or not Path("/mnt/datasets/public").exists(),
    reason="Requires presence of the benchmark and ImageNet.",
)
def test_toolbox_getitem():
    for benchmark_name in ("test", "imagenet", "ccb-test"):
        for task in io.task_iterator(benchmark_name):
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
