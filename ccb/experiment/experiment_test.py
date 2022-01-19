# TODO(drouin) finish adapting to pytest  (different than unittest)

import os
from ccb.experiment.experiment import get_model_generator, hparams_to_string


def test_trial_numbering():
    """
    Test that all hyperparameter combinations are numbered.

    """
    hp_str = list(zip(*hparams_to_string([{"key1": 1, "key2": 2}, {"key1": 2, "key2": 2}])))[1]
    assert all([x.startswith(f"trial_{i}") for i, x in enumerate(hp_str)])


def test_introspection():
    """
    Test if we correctly identify which hyperparameters are fixed vs. varying.

    """
    hp_str = list(zip(*hparams_to_string([{"key1": 1, "key2": 2}, {"key1": 2, "key2": 2}])))[1]
    assert all(["key1" in x for x in hp_str])
    assert all(["key2" not in x for x in hp_str])


def test_duplicate_combos():
    """
    Test if we correctly return two trials if we receive duplicate HP combinations.

    """
    hp_str = list(
        zip(*hparams_to_string([{"key1": 1, "key2": 2}, {"key1": 1, "key2": 2}, {"key1": 2, "key2": 3}]))
    )[1]
    assert len(hp_str) == 3
    assert len([x for x in hp_str if "key1=1" in x and "key2=2" in x]) == 2


def test_single_combo():
    """
    Test if we correctly handle the case where a single hyperparameter combination is received.
    Expected behavior is to have only the trial ID in the string.

    """
    hp_str = list(zip(*hparams_to_string([{"key1": 1, "key2": 2}])))[1]
    assert len(hp_str) == 1
    assert hp_str[0] == "trial_0"


def test_load_module():
    """
    Test loading an existing model generator from a user-specified path.

    """
    path = "tmp_testing_model_generator.py"
    open(path, "w").write("def model_generator():\n    return 'it works!'")
    assert get_model_generator(path)() == "it works!"
    os.remove(path)

# def test_unexisting_path():
#     """
#     Test trying to load from an unexisting module path.

#     """
#     self.assertRaises(ModuleNotFoundError, get_model_generator, "1234_unexisting_path.py")

# def test_missing_generator_instance():
#     """
#     Test trying to load when the module exists, but the variable is not defined.

#     """
#     path = "tmp_testing_model_generator_broken.py"
#     open(path, "w").write("def model_generator_():\n    pass")  # So model_generator doesn't exist
#     self.assertRaises(AttributeError, get_model_generator, path)
#     os.remove(path)
