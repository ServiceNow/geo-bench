#!/usr/bin/env python
"""
Trains the model using job information contained in the current directory.
Expects to find files "hparams.json" and "task_specs.json".

Usage: trainer.py --model-generator path/to/my/model/generator.py

"""
import argparse
import os

from ccb.torch_toolbox.dataset import DataModule
from ccb.experiment.experiment import get_model_generator, Job
import pytorch_lightning as pl


def start():
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="trainer.py",
        description="Trains the model using job information contained in the current directory.",
    )
    parser.add_argument(
        "--model-generator",
        help="Module name that defines a model generator. Must be in PYTHONPATH and expects a model_generator variable to exist.",
        required=True,
    )
    parser.add_argument(
        "--job-dir",
        help="Path to the job.",
        required=True,
    )
    args = parser.parse_args()

    job = Job(args.job_dir)
    hparams = job.hparams

    print("Model generator path:", args.model_generator)
    print("Hyperparameters:", hparams)
    print("Task specifications:", job.task_specs)

    # Load the user-specified model generator
    model_gen = get_model_generator(args.model_generator)
    model = model_gen.generate(job.task_specs, hparams)
    datamodule = DataModule(job.task_specs, batch_size=hparams["batch_size"], num_workers=hparams["num_workers"])
    # datamodule = Dataset(
    #     job.task_specs.dataset_name,
    #     os.environ.get("DATAROOT", str(job.dir.parent / "data")),
    #     job.task_specs,
    #     hparams,
    # )
    if hparams.get("logger", False) == "csv":
        logger = pl.loggers.CSVLogger(job.dir)
    else:
        logger = None
    trainer = pl.Trainer(gpus=0, max_epochs=1, max_steps=hparams.get("train_iters", None), logger=logger)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    start()
