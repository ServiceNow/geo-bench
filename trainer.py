"""
Trains the model using job information contained in the current directory.
Expects to find files "hparams.json" and "task_specs.json".

Usage: trainer.py --model-generator path/to/my/model/generator.py

"""
import argparse
import json
import os

from pathlib import Path
from uuid import uuid4

from toolbox import TaskSpecifications
from toolbox.dataset import Dataset, iter_datasets
from toolbox.model import ModelGenerator
from toolbox.utils import get_model_generator, hparams_to_string


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="trainer.py",
        description="Trains the model using job information contained in the current directory.",
    )
    parser.add_argument(
        "--model-generator",
        help="Path to a Python file that defines a model generator (expects a model_generator variable to exist).",
        required=True,
    )
    args = parser.parse_args()

    # Load the user-specified model generator
    model_generator = get_model_generator(args.model_generator)

    # Load hyperparameters and task specification
    hparams = json.load(open("hparams.json", "r"))
    task_specs = json.load(open("task_specs.json", "r"))
    print(hparams)
    print(task_specs)

    # TODO: Training
