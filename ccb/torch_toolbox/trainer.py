"""
Trains the model using job information contained in the current directory.
Expects to find files "hparams.json" and "task_specs.json".

Usage: trainer.py --model-generator path/to/my/model/generator.py

"""
import argparse
import json
import os
import pickle as pkl

from pathlib import Path
from uuid import uuid4

from ccb.io.task import TaskSpecifications
from ccb.torch_toolbox.dataset import Dataset
from ccb.torch_toolbox.model import ModelGenerator, Model
from ccb.experiment.experiment import get_model_generator, hparams_to_string
import pytorch_lightning as pl


def start():
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
    parser.add_argument(
        "--exp-dir",
        help="Path to the experiment",
        required=True,
    )
    args = parser.parse_args()

    # Load hyperparameters and task specification
    exp_dir = Path(args.exp_dir)
    with (exp_dir / "hparams.json").open("r") as fd:
        hparams = json.load(fd)
    with (exp_dir / "task_specifications.pkl").open("rb") as fd:
        task_specs = pkl.load(fd)

    print("Model generator path:", args.model_generator)
    print("Hyperparameters:", hparams)
    print("Task specifications:", task_specs)

    # Load the user-specified model generator
    model_gen = get_model_generator(args.model_generator)
    model = model_gen.generate(task_specs, hparams)
    datamodule = Dataset(
        task_specs.dataset_name, os.environ.get("DATAROOT", str(exp_dir.parent / "data")), task_specs, hparams
    )
    if hparams.get("logger", False) == "csv":
        logger = pl.loggers.CSVLogger(args.exp_dir)
    else:
        logger = None
    trainer = pl.Trainer(gpus=0, max_epochs=1, max_steps=hparams.get("train_iters", None), logger=logger)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    start()
