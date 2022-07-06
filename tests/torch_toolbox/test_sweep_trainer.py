"""Test sweep logic."""
import os
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path
from subprocess import PIPE

import pytest
from ruamel.yaml import YAML

from ccb.experiment.experiment import Job


def test_sweep():

    with open(
        os.path.join("tests", "data", "ccb-test-classification", "brick_kiln_v1.0", "task_specs.pkl"), "rb"
    ) as fd:
        task_specs = pickle.load(fd)

    yaml = YAML()

    with open(os.path.join("tests", "configs", "base_classification.yaml"), "r") as yamlfile:
        config = yaml.load(yamlfile)

    with open(os.path.join("tests", "configs", "classification_hparams.yaml"), "r") as yamlfile:
        hparams = yaml.load(yamlfile)

    with tempfile.TemporaryDirectory(prefix="test") as job_dir:
        job = Job(job_dir)
        task_specs.save(job.dir)

        job.save_hparams(hparams)
        job.save_config(config)

        sweep_name = "testSweep"

        shutil.copy(Path(config["wandb"]["sweep"]["sweep_config_path"]).resolve(), job_dir)
        sweep_config_path = os.path.join(job_dir, "sweep_config.yaml")
        with open(sweep_config_path, "r") as yamlfile:
            sweep_config = yaml.load(yamlfile)
        sweep_config["command"] = [  # commands needed to run actual training script
            "${program}",
            "--job_dir",
            str(job_dir),
        ]
        yaml = YAML()
        yaml.indent(sequence=4, offset=2)
        with open(sweep_config_path, "w") as yamlfile:
            yaml.dump(sweep_config, yamlfile)

        cmd = ["wandb", "sweep", "--name", sweep_name, sweep_config_path]

        result = subprocess.run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
        output = result.stderr.replace("\n", "")  # wandb sweep over stderr not stdout unexpectedly

        if "Run sweep agent with: " in output:
            wandb_agent_command = output.split("Run sweep agent with: ")[-1]
            sweep_id = wandb_agent_command.split(" ")[-1]
            config["wandb"]["sweep"]["sweep_id"] = sweep_id
            config["wandb"]["wandb_group"] = sweep_name
        else:
            raise ValueError(f"Sweep could not be launched successfully, got {output}")

        with open(os.path.join(job_dir, "config.yaml"), "w") as yamlfile:
            yaml.dump(config, yamlfile)

        import pdb

        pdb.set_trace()
        launch_agent_cmd = ["wandb", "agent", wandb_agent_command.split(" ")[-1], "--count", str(1)]
        result = subprocess.run(launch_agent_cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
        print(result.stderr.replace("\n", ""))
