from pathlib import Path
import subprocess
import sys
import tempfile
from ccb import io

import pytest
from ccb.experiment.experiment import Job, get_model_generator, hparams_to_string
from ccb.experiment.experiment_generator import experiment_generator
from ccb.experiment.sequential_dispatcher import sequential_dispatcher


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
    hp_str = list(zip(*hparams_to_string([{"key1": 1, "key2": 2}, {"key1": 1, "key2": 2}, {"key1": 2, "key2": 3}])))[1]
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

    model_generator = get_model_generator("ccb.torch_toolbox.model_generators.conv4")
    assert hasattr(model_generator, "hp_search")


def test_unexisting_path():
    """
    Test trying to load from an unexisting module path.

    """
    try:
        get_model_generator("ccb.torch_toolbox.model_generators.foobar")
    except Exception as e:
        assert isinstance(e, ModuleNotFoundError)


@pytest.mark.slow
def test_experiment_generator_on_mnist():

    with tempfile.TemporaryDirectory() as exp_dir:

        experiment_generator("ccb.torch_toolbox.model_generators.conv4", exp_dir, benchmark_name="test")

        sequential_dispatcher(exp_dir=exp_dir, prompt=False)

        for job_dir in (Path(exp_dir) / "MNIST").iterdir():
            job = Job(job_dir)
            print(job_dir)
            metrics = job.get_metrics()
            assert float(metrics["test_accuracy-1"]) > 0.10


@pytest.mark.slow
@pytest.mark.skipif(not Path(io.datasets_dir).exists(), reason="Requires presence of the benchmark.")
def test_experiment_generator_on_benchmark():
    experiment_generator_dir = Path(__file__).absolute().parent

    experiment_dir = tempfile.mkdtemp(prefix="exp_gen_test_on_benchmark")
    # experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating experiments in {experiment_dir}.")
    cmd = [
        sys.executable,
        str(experiment_generator_dir / "experiment_generator.py"),
        "--model-generator",
        "ccb.torch_toolbox.model_generators.conv4_test",
        "--experiment-dir",
        str(experiment_dir),
        "--benchmark",
        "ccb-test",
    ]
    subprocess.check_call(cmd)

    sequential_dispatcher(exp_dir=experiment_dir, prompt=False)
    for ds_dir in Path(experiment_dir).iterdir():
        for job_dir in ds_dir.iterdir():
            job = Job(job_dir)
            print(job_dir)
            metrics = job.get_metrics()
            print(metrics)


if __name__ == "__main__":
    test_experiment_generator_on_benchmark()
