import subprocess
import sys
from pathlib import Path
import tempfile

from ccb.experiment.experiment import Job
from ccb.io import mnist_task_specs
from ccb.torch_toolbox.model_generators import conv4


def test_toolbox_mnist():

    with tempfile.TemporaryDirectory(prefix="ccb_mnist_test") as job_dir:
        job = Job(job_dir)
        mnist_task_specs.save(job.dir)

        hparams = conv4.model_generator.hp_search(mnist_task_specs)[0]
        job.save_hparams(hparams)

        torch_toolbox_dir = Path(__file__).absolute().parent.parent
        cmd = [
            sys.executable,
            str(torch_toolbox_dir / "trainer.py"),
            "--model-generator",
            "ccb.torch_toolbox.model_generators.conv4",
            "--job-dir",
            job.dir,
        ]
        subprocess.run(cmd)
        print(job.metrics)
        assert float(job.metrics["train_acc1_step"]) > 20  # has to be better than random after seeing 20 batches


if __name__ == "__main__":
    test_toolbox_mnist()
