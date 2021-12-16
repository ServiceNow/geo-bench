from importlib import import_module
from itertools import chain


def get_model_generator(path):
    """
    Parameters:
    -----------
    path: str
        The path of the model generator module.

    Returns:
    --------
    model_generator: a model_generator function loaded from the module.

    """
    # Preprocess path
    model_generator_path = path.replace(".py", "")  # Need module name, not file

    # Load user-provided module
    return import_module(model_generator_path).model_generator


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
    def _format_combo(hps):
        return "_".join(f"{k}={hps[k]}" for k in active_keys)

    return [(hps, _format_combo(hps)) for hps in hp_configs]
