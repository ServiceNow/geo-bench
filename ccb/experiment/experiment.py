"""Experiment."""

import csv
import glob
import json
import os
import pickle
import stat
import sys
from functools import cached_property
from importlib import import_module
from itertools import chain
from os import mkdir
from pathlib import Path
from typing import Any, Dict, List, Union

from ruamel.yaml import YAML

from ccb import io
from ccb.torch_toolbox.model import ModelGenerator


def get_model_generator(module_name: str, hparams: Dict[str, Any] = {}) -> ModelGenerator:
    """Return the model generator module based on name with a set of hyperparameters.

    Args:
        module_name: The module_name of the model generator module.
        hparams: hparameter dict to overwrite the default base values

    Returns:
        a model_generator function loaded from the module with hparams
    """
    return import_module(module_name).model_generator(hparams)


class Job:
    """Job.

    Helper class to organize running of experiments.
    """

    def __init__(self, dir: str) -> None:
        """Initialize new instance of Job.

        Args:
            dir: path to directory where job should be created
        """
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    @cached_property
    def hparams(self):
        """Return hyperparameters."""
        with open(self.dir / "hparams.json") as fd:
            return json.load(fd)

    def save_hparams(
        self, hparams: Dict[str, Union[str, float, int, List[int], List[str]]], overwrite: bool = False
    ) -> None:
        """Save hyperparameters in job directory.

        Args:
            hparams: set of hyperparameters to save
            overwrite: whether to overwrite existing hparams
        """
        hparams_path = self.dir / "hparams.json"
        if hparams_path.exists() and not overwrite:
            raise Exception("hparams alread exists and overwrite is set to False.")
        with open(hparams_path, "w") as fd:
            json.dump(hparams, fd, indent=4, sort_keys=True)
            self.hparams = hparams

    @cached_property
    def task_specs(self):
        """Return task specifications."""
        with open(self.dir / "task_specs.pkl", "rb") as fd:
            return pickle.load(fd)

    def get_metrics(self):
        """Retrieve the metrics after training from job directory."""
        if self.hparams.get("logger", "") == "wandb":
            import wandb

            wandb.finish()
            summary = glob.glob(str(self.dir / "wandb" / "latest-run" / "*" / "wandb-summary.json"))
            with open(summary[0], "r") as infile:
                data = json.load(infile)
            return data
        else:
            try:
                with open(self.dir / "lightning_logs" / "version_0" / "metrics.csv", "r") as fd:
                    data = {}
                    # FIXME: This would be more efficient if done backwards
                    for entry in csv.DictReader(fd):
                        data.update({k: v for k, v in entry.items() if v != ""})
                return data
            except FileNotFoundError as e:
                stderr = self.get_stderr()
                if stderr is not None:
                    raise Exception(stderr)
                else:
                    raise e

    def save_task_specs(self, task_specs: io.TaskSpecifications, overwrite: bool = False) -> None:
        """Save task specifications in job directory.

        Args:
            task_specs: task specifications
            overwrite: whether to overwrite existing task specs
        """
        task_specs.save(self.dir, overwrite=overwrite)

    def write_script(self, model_generator_module_name: str, job_dir: str) -> None:
        """Write bash scrip that can be executed to run job.

        Args:
            model_generator_module_name: what model_generator to use
            job_dir: job directory from which to run job
        """
        script_path = self.dir / "run.sh"
        with open(script_path, "w") as fd:
            fd.write("#!/bin/bash\n")
            fd.write("# Usage: sh run.sh path/to/model_generator.py\n\n")
            fd.write(
                f"ccb-trainer --model-generator {model_generator_module_name} --job-dir {job_dir} >log.out 2>err.out"
            )
        script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

    def write_wandb_sweep_cl_script(
        self, model_generator_module_name: str, job_dir: str, base_sweep_config: str
    ) -> None:
        """Write final sweep_config.yaml that can be used to initialize sweep.

        Args:
            model_generator_module_name: what model_generator to use
            job_dir: job directory from which to run job
            base_sweep_config: path to base sweep config yaml file for wandb
        """
        yaml = YAML()
        with open(base_sweep_config, "r") as yamlfile:
            base_yaml = yaml.load(yamlfile)  # Note the safe_load

        base_yaml["command"] = [  # commands needed to run actual training script
            "${program}",
            "--model-generator",
            model_generator_module_name,
            "--job-dir",
            str(job_dir),
        ]

        # sweep name that will be seen on wandb
        if model_generator_module_name != "ccb.torch_toolbox.model_generators.py_segmentation_generator":
            backbone = get_model_generator(model_generator_module_name).base_hparams["backbone"]
            base_yaml["name"] = "_".join(str(job_dir).split("/")[-2:]) + "_" + backbone
        else:
            encoder = get_model_generator(model_generator_module_name).base_hparams["encoder_type"]
            decoder = get_model_generator(model_generator_module_name).base_hparams["decoder_type"]
            base_yaml["name"] = "_".join(str(job_dir).split("/")[-2:]) + "_" + encoder + "_" + decoder

        save_path = os.path.join(job_dir, "sweep_config.yaml")
        yaml.indent(sequence=4, offset=2)
        with open(save_path, "w") as yamlfile:
            yaml.dump(base_yaml, yamlfile)

    def get_stderr(self) -> Union[str, None]:
        """Return error output from executing script."""
        try:
            with open(self.dir / "err.out", "r") as fd:
                return fd.read()
        except FileNotFoundError:
            return None

    def get_stdout(self) -> str:
        """Return log output from executing script."""
        with open(self.dir / "log.out", "r") as fd:
            return fd.read()
