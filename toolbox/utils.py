import sys

from importlib import import_module
from itertools import chain
from pathlib import Path


def get_model_generator(path: str):
    """
    Parameters:
    -----------
    path: str
        The path of the model generator module.

    Returns:
    --------
    model_generator: a model_generator function loaded from the module.

    """
    path = Path(path)

    # Add the module to the PYTHONPATH
    sys.path.append(str(path.parent))

    # Load the module and extract the model generator
    return import_module(path.name.replace(".py", "")).model_generator


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
    active_keys = [k for k in keys if len(set(combo[k] for combo in hp_configs)) > 1]

    # Pretty print a HP combination
    def _format_combo(trial_id, hps):
        # XXX: we include a trial_id prefix to deal with duplicate combinations or the case where active_keys is empty
        return f"trial_{trial_id}" + (
            "__" + "_".join(f"{k}={hps[k]}" for k in active_keys) if len(active_keys) > 0 else ""
        )

    # XXX: append i
    return [(hps, _format_combo(i, hps)) for i, hps in enumerate(hp_configs)]
