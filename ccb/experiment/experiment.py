import csv
import json
from os import mkdir
import pickle
import stat
import sys
import glob
from importlib import import_module
from itertools import chain
from pathlib import Path
from functools import cached_property
from ccb import io
from ccb.torch_toolbox.model import ModelGenerator
from ruamel.yaml import YAML
import os
from typing import Dict, Any


def get_model_generator(module_name: str, hparams: Dict[str, Any] = {}) -> ModelGenerator:
    """
    Parameters:
    -----------
    module_name: str
        The module_name of the model generator module.
    hparams:
        hparameter dict to overwrite the default base values
        
    Returns:
    --------
    model_generator: a model_generator function loaded from the module.
    """

    return import_module(module_name).model_generator(hparams)


def hparams_to_string(hp_configs):
    """
    Generate a string respresentation of the meaningful hyperparameters. This string will be used for file names and
    job names, to be able to distinguish them easily.
    Parameters:
    -----------
    hp_configs: list of dicts
        A list of dictionnaries that each contain one hyperparameter configuration
    Returns:
    --------
    A list of pairs of hyperparameter combinations (dicts from the input, string representation)
    """
    # Find which hyperparameters vary between hyperparameter combinations
    keys = set(chain.from_iterable(combo.keys() for combo in hp_configs))

    # TODO find a solution for unhashable hparams such as list, or print a more
    # useful error message.
    # active_keys = [k for k in keys if len(set(combo[k] for combo in hp_configs)) > 1]
    active_keys = []
    # assuming that all the dicts in hp_configs have the same keys
    hp_config_keys = [[key for key in combo] for combo in hp_configs]

    for keys in zip(*hp_config_keys):
        first_val = hp_configs[0][keys[0]]
        for idx, key in enumerate(keys):
            val = hp_configs[idx][keys[idx]]
            if first_val != val:
                active_keys.append(key)

    # Pretty print a HP combination
    def _format_combo(trial_id, hps):
        # XXX: we include a trial_id prefix to deal with duplicate combinations or the case where active_keys is empty
        return f"trial_{trial_id}" + (
            "__" + "_".join(f"{k}={hps[k]}" for k in active_keys) if len(active_keys) > 0 else ""
        )

    return [(hps, _format_combo(i, hps)) for i, hps in enumerate(hp_configs)]


class Job:
    def __init__(self, dir) -> None:
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    @cached_property
    def hparams(self):
        with open(self.dir / "hparams.json") as fd:
            return json.load(fd)

    def save_hparams(self, hparams, overwrite=False):
        hparams_path = self.dir / "hparams.json"
        if hparams_path.exists() and not overwrite:
            raise Exception("hparams alread exists and overwrite is set to False.")
        with open(hparams_path, "w") as fd:
            json.dump(hparams, fd, indent=4, sort_keys=True)
            self.hparams = hparams

    @cached_property
    def task_specs(self):
        with open(self.dir / "task_specs.pkl", "rb") as fd:
            return pickle.load(fd)

    def get_metrics(self):
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

    def save_task_specs(self, task_specs: io.TaskSpecifications, overwrite=False):
        task_specs.save(self.dir, overwrite=overwrite)

    def write_script(self, model_generator_module_name: str, job_dir: str, wandb_mode: str):
        """Write bash scrip that can be executed to run job.
        Args:
            model_generator_module_name: what model_generator to use
            job_dir: job directory from which to run job
            wandb_mode: wandb_mode: what kind of experiment to dispatch, ["sweep", "seeded_runs", "standard"]

        """
        script_path = self.dir / "run.sh"
        with open(script_path, "w") as fd:
            fd.write("#!/bin/bash\n")
            fd.write("# Usage: sh run.sh path/to/model_generator.py\n\n")
            fd.write(
                f'cd $(dirname "$0") && ccb-trainer --model-generator {model_generator_module_name} --job-dir {job_dir} --wandb-mode {wandb_mode} >log.out 2>err.out'
            )
        script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

    def write_wandb_sweep_cl_script(self, model_generator_module_name: str, job_dir: str, base_sweep_config: str):
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
            "--wandb-mode",
            "sweep",
        ]

        # sweep name that will be seen on wandb
        backbone = get_model_generator(model_generator_module_name).base_hparams["backbone"]
        base_yaml["name"] = "_".join(str(job_dir).split("/")[-2:]) + "_" + backbone

        save_path = os.path.join(job_dir, "sweep_config.yaml")
        yaml.indent(sequence=4, offset=2)
        with open(save_path, "w") as yamlfile:
            yaml.dump(base_yaml, yamlfile)

    def get_stderr(self):
        try:
            with open(self.dir / "err.out", "r") as fd:
                return fd.read()
        except FileNotFoundError:
            return None

    def get_stdout(self):
        with open(self.dir / "log.out", "r") as fd:
            return fd.read()
