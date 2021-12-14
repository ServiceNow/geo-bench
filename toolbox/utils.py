from itertools import chain


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
