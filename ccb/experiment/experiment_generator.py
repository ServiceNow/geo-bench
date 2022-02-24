"""
Generate experiment directory structure

Usage: experiment_generator.py --model-generator path/to/my/model/generator.py  --experiment-dir path/to/my/experiments

"""
import argparse

from pathlib import Path
from uuid import uuid4
from typing import Iterator

from ccb.experiment.experiment import Job
from ccb.torch_toolbox.model import ModelGenerator
from ccb.experiment.experiment import get_model_generator
from ccb import io


def experiment_generator(
    model_generator_module_name: str,
    experiment_dir: str,
    task_filter: callable = None,
    max_num_configs: int = 10,
    benchmark_name="default",
):
    """
    Generates the directory structure for every tasks and every hyperparameter configuration.
    According to model_generator.hp_search.

    Parameters:
    -----------
    model_generator: ModelGenerator
        The generator associated with the current model. Used to get hyperparameter combinations.
    experiment_dir: str
        The directory in which to create the experiment directories.
    task_filter: callable(TaskSpecification)
        A function that takes as input a task specification instance and returns False if it should be skipped.

    """
    experiment_dir = Path(experiment_dir)
    experiment_dir /= str(uuid4())

    model_generator = get_model_generator(model_generator_module_name)

    for task_specs in io.task_iterator(benchmark_name=benchmark_name):
        if task_filter is not None:
            if not task_filter(task_specs):
                continue

        for hparams, hparams_string in model_generator.hp_search(task_specs, max_num_configs):

            # Create and fill experiment directory
            job_dir = experiment_dir / task_specs.dataset_name / hparams_string
            job = Job(job_dir)
            job.save_hparams(hparams)
            job.save_task_specs(task_specs)
            job.write_script(model_generator_module_name)


def start():
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="experiment_generator.py",
        description="Generate experiment directory structure based on user-defined model generator",
    )
    parser.add_argument(
        "--model-generator",
        help="Path to a Python file that defines a model generator (expects a model_generator variable to exist).",
        required=True,
    )
    parser.add_argument(
        "--experiment-dir",
        help="The based directory in which experiment-related files should be created.",
        required=True,
    )

    parser.add_argument(
        "--benchmark",
        help="The set of dataset that will be used for evaluating. 'ccb' | 'mnist' ",
        required=False,
        default="default",
    )

    args = parser.parse_args()

    # Generate experiments
    experiment_generator(args.model_generator, args.experiment_dir, benchmark_name=args.benchmark)


if __name__ == "__main__":
    start()
