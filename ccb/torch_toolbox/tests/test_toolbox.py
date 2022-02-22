import json
import subprocess
import sys
import os
import csv
import shutil as sh
from pathlib import Path

from ccb.experiment.experiment import get_model_generator
from ccb.io.task import Classification, TaskSpecifications
from ccb.torch_toolbox.model import head_generator, train_loss_generator
from ccb.torch_toolbox.trainer import start


def test_toolbox_mnist():
    hyperparameters = {
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
    test_dir = "/tmp/cc_benchmark/mnist_test"
    sh.rmtree(test_dir, ignore_errors=True)
    os.makedirs(test_dir, exist_ok=True)
    with open(Path(test_dir) / "hparams.json", "w") as fd:
        json.dump(hyperparameters, fd)

    specs = TaskSpecifications(patch_size=(28, 28, 1, 1), label_type=Classification(10), dataset_name="MNIST")
    specs.save(test_dir)
    python_binary = sys.executable
    train_path = Path(__file__).absolute().parent.parent / "trainer.py"
    model_path = Path(__file__).absolute().parent.parent / "model_generators" / "conv4.py"
    cmd = [
        python_binary,
        str(train_path),
        "--model-generator",
        str(model_path),
        "--exp-dir",
        test_dir,
    ]
    subprocess.run(cmd)
    with open(Path(test_dir) / "default/version_0/metrics.csv", "r") as fd:
        data = next(csv.DictReader(fd))
    print(data)
    assert float(data["train_acc1_step"]) > 20  # has to be better than random after seeing 20 batches


if __name__ == "__main__":
    test_toolbox_mnist()
