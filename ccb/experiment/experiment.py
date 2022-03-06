import csv
import json
from os import mkdir
import pickle
import stat
import sys

from importlib import import_module
from itertools import chain
from pathlib import Path
from functools import cached_property
from ccb import io
from ccb.torch_toolbox.model import ModelGenerator


def get_model_generator(module_name: str) -> ModelGenerator:
    """
    Parameters:
    -----------
    module_name: str
        The module_name of the model generator module.

    Returns:
    --------
    model_generator: a model_generator function loaded from the module.

    """

    return import_module(module_name).model_generator


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
    active_keys = [k for k in keys if len(set(combo[k] for combo in hp_configs)) > 1]

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
            json.dump(hparams, fd)
            self.hparams = hparams

    @cached_property
    def metrics(self):
        with open(self.dir / "default" / "version_0" / "metrics.csv", "r") as fd:
            data = next(csv.DictReader(fd))
        return data

    @cached_property
    def task_specs(self):
        with open(self.dir / "task_specs.pkl", "rb") as fd:
            return pickle.load(fd)

    def save_task_specs(self, task_specs: io.TaskSpecifications, overwrite=False):
        task_specs.save(self.dir, overwrite=overwrite)

    def write_script(self, model_generator_module):
        script_path = self.dir / "run.sh"
        with open(script_path, "w") as fd:
            fd.write("#!/bin/bash\n")
            fd.write("# Usage: sh run.sh path/to/model_generator.py\n\n")
            fd.write(
                f'cd $(dirname "$0") && ccb-trainer --model-generator {model_generator_module} --job-dir . >log.out 2>err.out'
            )
        script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

    @cached_property
    def stderr(self):
        with open(self.dir / "err.out", "r") as fd:
            return fd.read()

    @cached_property
    def stdout(self):
        with open(self.dir / "log.out", "r") as fd:
            return fd.read()
