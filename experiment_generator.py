"""
Generate experiment directory structure

Arguments: a user-defined model generator

Usage: experiment_generator.py path/to/my/model/example_model_genartor.py

"""
import argparse
import json
import os

from pathlib import Path
from uuid import uuid4

from toolbox.dataset import Dataset, iter_datasets
from toolbox.model import ModelGenerator
from toolbox.utils import get_model_generator, hparams_to_string


# def _raise_error(msg):
#     print(msg)
#     exit(1)


# def get_model_generator(path):
#     """
#     Parameters:
#     -----------
#     path: str
#         The path of the model generator module.

#     Returns:
#     --------
#     model_generator: a model_generator function loaded from the module.

#     """
#     # Preprocess path
#     model_generator_path = path.replace(".py", "")  # Need module name, not file

#     # Try to load user-provided module
#     try:
#         model_generator = importlib.import_module(model_generator_path).model_generator
#     except AttributeError:
#         _raise_error(
#             f"Error: The model generator ({model_generator_path}) does not contain a 'model_generator' function."
#         )
#     except ModuleNotFoundError:
#         _raise_error(f"Error: The model generator module ({model_generator_path}) cannot be found.")

#     return model_generator


def experiment_generator(
    model_generator: ModelGenerator,
    model_generator_path: str,
    experiment_dir: str,
    task_filter: callable = None,
    max_num_configs: int = 10,
):
    """
    Generates the directory structure for every tasks and every hyperparameter configuration.
    According to model_generator.hp_search.

    """
    experiment_dir = Path(experiment_dir)
    experiment_dir /= str(uuid4())

    # TODO create experiment directory and append date in the dir name.
    for dataset in iter_datasets():
        if task_filter is not None:
            if not task_filter(dataset.task_specs):
                continue

        for hyperparams, hyperparams_string in model_generator.hp_search(dataset.task_specs, max_num_configs):

            # Create experiment directory
            path = experiment_dir / dataset.name / hyperparams_string
            os.makedirs(path, exist_ok=False)

            # Dump HPs
            hp_path = path / "hps.json"
            json.dump(hyperparams, open(hp_path, "w"))

            # Dump task specification files
            with open(path / "run.sh", "w") as f_cmd:
                f_cmd.write("#!/bin/bash\n")

                for i, task in enumerate(dataset.task_specs):

                    # Write task specification
                    task_path = path / f"task_{i}.json"
                    json.dump(task.to_dict(), open(task_path, "w"))

                    # Add command to the launcher
                    f_cmd.write(
                        f"python trainer.py --model-generator {model_generator_path} --hps {hp_path} --task-spec {task_path}\n"
                    )


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="experiment_generator.py",
        description="Generate experiment directory structure based on user-defined model generator",
    )
    parser.add_argument("--model-generator")
    parser.add_argument("--experiment-dir")
    args = parser.parse_args()

    # Load the user-specified model generator
    model_generator = get_model_generator(args.model_generator)

    # Generate experiments
    experiment_generator(model_generator, args.model_generator, args.experiment_dir)
