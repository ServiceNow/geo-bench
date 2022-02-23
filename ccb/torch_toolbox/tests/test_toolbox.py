import subprocess
import sys
from pathlib import Path
import tempfile

from ccb.experiment.experiment import Job
from ccb.io import Classification, TaskSpecifications


def test_toolbox_mnist():
    hparams = {
        "lr_milestones": [10, 20],
        "lr_gamma": 0.1,
        "lr_backbone": 1e-3,
        "lr_head": 1e-3,
        "head_type": "linear",
        "train_iters": 50,
        "features_shape": (64,),
        "loss_type": "crossentropy",
        "batch_size": 64,
        "num_workers": 4,
        "logger": "csv",
    }

    with tempfile.TemporaryDirectory(prefix="ccb_mnist_test") as job_dir:
        job = Job(job_dir)
        job.save_hparams(hparams)

        specs = TaskSpecifications(patch_size=(28, 28, 1, 1), label_type=Classification(10), dataset_name="MNIST")
        specs.save(job.dir)

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
