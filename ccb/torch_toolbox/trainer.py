#!/usr/bin/env python
"""
Trains the model using job information contained in the current directory.
Expects to find files "hparams.json" and "task_specs.json".

Usage: trainer.py --model-generator path/to/my/model/generator.py

"""
import argparse

from ccb.torch_toolbox.dataset import DataModule
from ccb.experiment.experiment import get_model_generator, Job
import pytorch_lightning as pl


def train(model_gen, job_dir):
    job = Job(job_dir)
    hparams = job.hparams

    model = model_gen.generate(job.task_specs, hparams)
    datamodule = DataModule(
        job.task_specs,
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        transform=model_gen.get_transform(job.task_specs, hparams),
        collate_fn=model_gen.get_collate_fn(job.task_specs, hparams),
    )

    loggers = [pl.loggers.CSVLogger(str(job.dir))]
    if hparams.get("logger", "").lower() == "wandb":
        loggers.append(pl.loggers.WandbLogger(project="ccb", name=hparams.get("name", str(job.dir)), save_dir=str(job.dir)))
    trainer = pl.Trainer(
        gpus=hparams.get("n_gpus", 1),
        max_epochs=hparams["max_epochs"],
        max_steps=hparams.get("train_iters", None),
        limit_val_batches=hparams.get("limit_val_batches", 1.0),
        limit_test_batches=hparams.get("limit_val_batches", 1.0),
        val_check_interval=hparams.get("val_check_interval", 1.0),
        accelerator=hparams.get("accelerator", None),
        progress_bar_refresh_rate=0,
        logger=loggers,
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


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

    # Load the user-specified model generator
    model_gen = get_model_generator(args.model_generator)
    train(model_gen, args.job_dir)


if __name__ == "__main__":
    start()
