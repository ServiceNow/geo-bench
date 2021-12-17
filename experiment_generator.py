"""
Generate experiment directory structure

Usage: experiment_generator.py --model-generator path/to/my/model/generator.py  --experiment-dir path/to/my/experiments

"""
import argparse
import json
import os

from pathlib import Path
from uuid import uuid4

from toolbox.dataset import Dataset, iter_datasets
from toolbox.model import ModelGenerator
from toolbox.utils import get_model_generator, hparams_to_string


TRAINER_CMD = "python trainer.py"


def experiment_generator(
    model_generator: ModelGenerator,
    experiment_dir: str,
    task_filter: callable = None,
    max_num_configs: int = 10,
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

    for dataset in iter_datasets():
        if task_filter is not None:
            if not task_filter(dataset.task_specs):
                continue

        for hparams, hparams_string in model_generator.hp_search(dataset.task_specs, max_num_configs):

            # Create experiment directory
            job_dir = experiment_dir / dataset.name / hparams_string
            job_dir.mkdir(parents=True, exist_ok=False)

            # Dump HPs
            json.dump(hparams, open(job_dir / "hparams.json", "w"))

            # Dump task specification
            json.dump(dataset.task_specs.to_dict(), open(job_dir / "task_specs.json", "w"))

            # Experiment launch file
            with open(job_dir / "run.sh", "w") as f_cmd:
                f_cmd.write("#!/bin/bash\n")
                f_cmd.write(f'cd $(dirname "$0"); {TRAINER_CMD} >log.out 2>err.out')


if __name__ == "__main__":
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
    args = parser.parse_args()

    # Load the user-specified model generator
    model_generator = get_model_generator(args.model_generator)

    # Generate experiments
    experiment_generator(model_generator, args.experiment_dir)
