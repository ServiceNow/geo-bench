import pickle
import tempfile
from ccb.experiment.experiment import Job
from ccb.io import mnist_task_specs
from ccb.torch_toolbox.model_generators import conv4
from ccb.torch_toolbox import trainer
from ccb import io
from pathlib import Path


def train_job_on_task(model_generator, task_specs):
    with tempfile.TemporaryDirectory(prefix="ccb_mnist_test") as job_dir:
        job = Job(job_dir)
        task_specs.save(job.dir)

        hparams = model_generator.hp_search(task_specs)[0][0]
        job.save_hparams(hparams)

        trainer.train(model_gen=model_generator, job_dir=job_dir)
        hparams = job.hparams

        metrics = job.get_metrics()
        print(metrics)
        assert float(metrics["train_acc1_step"]) > 20  # has to be better than random after seeing 20 batches


def test_toolbox_mnist():
    train_job_on_task(conv4.model_generator, mnist_task_specs)


def test_toolbox_brick_kiln():
    with open(Path(io.datasets_dir) / "brick_kiln_v1.0" / "task_specs.pkl", "rb") as fd:
        task_specs = pickle.load(fd)
    train_job_on_task(conv4.model_generator, task_specs)


if __name__ == "__main__":
    test_toolbox_brick_kiln()
