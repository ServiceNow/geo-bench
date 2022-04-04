import pickle
import tempfile

import pytest
from ccb.experiment.experiment import Job
from ccb.io import mnist_task_specs
from ccb.torch_toolbox.model_generators import conv4_test, timm_generator
from ccb.torch_toolbox import trainer
from ccb import io
from pathlib import Path


def train_job_on_task(model_generator, task_specs, threshold, logger=None):
    with tempfile.TemporaryDirectory(prefix="ccb_mnist_test") as job_dir:
        job = Job(job_dir)
        task_specs.save(job.dir)

        hparams = model_generator.hp_search(task_specs)[0][0]
        if logger is not None:
            hparams["logger"] = logger
        job.save_hparams(hparams)

        trainer.train(model_gen=model_generator, job_dir=job_dir)
        hparams = job.hparams

        metrics = job.get_metrics()
        print(metrics)
        assert float(metrics["test_accuracy-1"]) > threshold  # has to be better than random after seeing 20 batches


@pytest.mark.slow
def test_toolbox_mnist():
    train_job_on_task(conv4_test.model_generator, mnist_task_specs, 0.05)


@pytest.mark.optional
def test_toolbox_wandb():
    train_job_on_task(conv4_test.model_generator, mnist_task_specs, 0.05, logger="wandb")


@pytest.mark.slow
@pytest.mark.skipif(not Path(io.CCB_DIR).exists(), reason="Requires presence of the benchmark.")
def test_toolbox_brick_kiln():
    with open(Path(io.CCB_DIR) / "ccb-test" / "brick_kiln_v1.0" / "task_specs.pkl", "rb") as fd:
        task_specs = pickle.load(fd)
    train_job_on_task(conv4_test.model_generator, task_specs, 0.70)


@pytest.mark.slow
@pytest.mark.skipif(not Path(io.CCB_DIR).exists(), reason="Requires presence of the benchmark.")
def test_toolbox_timm():
    with open(Path(io.CCB_DIR) / "ccb-test" / "brick_kiln_v1.0" / "task_specs.pkl", "rb") as fd:
        task_specs = pickle.load(fd)
    train_job_on_task(timm_generator.model_generator, task_specs, 0.70)


if __name__ == "__main__":
    # test_toolbox_brick_kiln()
    # test_toolbox_wandb()
    test_toolbox_mnist()
    # test_toolbox_timm()
