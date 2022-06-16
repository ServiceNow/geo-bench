import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import ccb
from ccb import io
from ccb.experiment.experiment import Job, get_model_generator
from ccb.experiment.experiment_generator import experiment_generator
from ccb.experiment.sequential_dispatcher import sequential_dispatcher
from ccb.torch_toolbox.trainer import train


def test_load_module():
    """
    Test loading an existing model generator from a user-specified path.

    """

    model_generator = get_model_generator("ccb.torch_toolbox.model_generators.conv4")
    assert hasattr(model_generator, "generate")


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

        job = Job(Path(exp_dir) / "MNIST")
        print(Path(exp_dir) / "MNIST")
        metrics = job.get_metrics()
        assert float(metrics["test_Accuracy"]) > 0.05


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_gen, benchmark_name",
    [
        ("ccb.torch_toolbox.model_generators.conv4", "ccb-test-classification"),
        ("ccb.torch_toolbox.model_generators.py_segmentation_generator", "ccb-test-segmentation"),
    ],
)
def test_experiment_generator_on_benchmark(model_gen, benchmark_name):
    experiment_generator_dir = Path(ccb.experiment.__file__).absolute().parent

    experiment_dir = tempfile.mkdtemp(prefix="exp_gen_test_on_benchmark")

    print(f"Generating experiments in {experiment_dir}.")
    cmd = [
        sys.executable,
        str(experiment_generator_dir / "experiment_generator.py"),
        "--model-generator",
        model_gen,
        "--experiment-dir",
        str(experiment_dir),
        "--benchmark",
        benchmark_name,
    ]

    subprocess.check_call(cmd)

    sequential_dispatcher(exp_dir=experiment_dir, prompt=False, env=dict(os.environ))
    for ds_dir in Path(experiment_dir).iterdir():
        job = Job(ds_dir)
        print(ds_dir)
        metrics = job.get_metrics()
        print(metrics)


if __name__ == "__main__":
    test_experiment_generator_on_benchmark()
