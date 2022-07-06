"""Test trainer.py"""

import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

from ruamel.yaml import YAML

import ccb
from ccb.experiment.experiment import Job


def test_trainer_start():
    with open(
        os.path.join("tests", "data", "ccb-test-segmentation", "cvpr_chesapeake_landcover", "task_specs.pkl"), "rb"
    ) as fd:
        task_specs = pickle.load(fd)

    yaml = YAML()
    with open(os.path.join("tests", "configs", "base_segmentation.yaml"), "r") as yamlfile:
        config = yaml.load(yamlfile)

    with open(os.path.join("tests", "configs", "segmentation_hparams.yaml"), "r") as yamlfile:
        hparams = yaml.load(yamlfile)

    toolbox_dir = Path(ccb.torch_toolbox.__file__).absolute().parent

    with tempfile.TemporaryDirectory(prefix="test") as job_dir:
        # job_dir = f"{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"
        # os.makedirs(job_dir, exist_ok=True)

        job = Job(job_dir)
        task_specs.save(job.dir)

        job.save_hparams(hparams)
        job.save_config(config)

        cmd = [sys.executable, str(toolbox_dir / "trainer.py"), "--job_dir", job_dir]
        subprocess.call(cmd)
