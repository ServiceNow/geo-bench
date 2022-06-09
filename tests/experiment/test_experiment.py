from pathlib import Path
import subprocess
import sys
import tempfile
from ccb import io

import pytest
from ccb.experiment.experiment import Job, get_model_generator
from ccb.experiment.experiment_generator import experiment_generator
from ccb.experiment.sequential_dispatcher import sequential_dispatcher


def test_load_module():
    """
    Test loading an existing model generator from a user-specified path.

    """

    model_generator = get_model_generator("ccb.torch_toolbox.model_generators.conv4_test")
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

        experiment_generator("ccb.torch_toolbox.model_generators.conv4_test", exp_dir, benchmark_name="test")

        sequential_dispatcher(exp_dir=exp_dir, prompt=False)

        job = Job(Path(exp_dir) / "MNIST")
        print(Path(exp_dir) / "MNIST")
        metrics = job.get_metrics()
        assert float(metrics["test_Accuracy"]) > 0.05


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
        "ccb-test-small",
    ]

    subprocess.check_call(cmd)

    sequential_dispatcher(exp_dir=experiment_dir, prompt=False)
    for ds_dir in Path(experiment_dir).iterdir():
        job = Job(ds_dir)
        print(ds_dir)
        metrics = job.get_metrics()
        print(metrics)


if __name__ == "__main__":
    test_experiment_generator_on_benchmark()
