#!/usr/bin/env python
"""Wandb sweep the model using job information contained in the current directory."""

import argparse
import os

import pytorch_lightning as pl
import wandb

from ccb.experiment.experiment import Job, get_model_generator
from ccb.torch_toolbox.dataset import DataModule


def train(job_dir) -> None:
    """Train a model from the model generator on datamodule.

    Args:
        job_dir: job directory that contains task_specs and hparams.json

    """
    job = Job(job_dir)
    hparams = job.hparams
    config = job.config
    task_specs = job.task_specs
    seed = config["model"].get("seed", None)
    if seed is not None:
        pl.seed_everything(seed, workers=True)

    # Load the user-specified model generator
    model_gen = get_model_generator(config["model"]["model_generator_module_name"])

    with wandb.init(
        dir=job_dir,
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        group=config["wandb"].get("wandb_group", None),
        allow_val_change=True,
    ) as run:

        wandb_config = run.config
        # set up W&B logger
        # wandb_logger = pl.loggers.WandbLogger(
        #     project="ccb",
        #     entity="climate-benchmark",
        #     id=None,
        #     group=wandb_config.get("wandb_group", None),
        #     name=wandb_config.get("name", None),
        #     save_dir=str(job.dir),
        #     resume=True,
        # )

        # loggers = [pl.loggers.CSVLogger(str(job.dir), name="csv_logs"), wandb_logger]

        print(wandb.config)
        # instantiate model - need to used wandb config hparams here that the sweep overwrites
        model = model_gen.generate_model(task_specs=job.task_specs, hparams=wandb_config, config=config)

        trainer = model_gen.generate_model(config=config, hparams=wandb_config, job=job)

        # reload config
        config = job.config

        datamodule = DataModule(
            task_specs,
            benchmark_dir=config["experiment"]["benchmark_dir"],
            batch_size=hparams["batch_size"],
            num_workers=config["dataloader"]["num_workers"],
            train_transform=model_gen.get_transform(task_specs=task_specs, hparams=hparams, config=config, train=True),
            eval_transform=model_gen.get_transform(task_specs=task_specs, hparams=hparams, config=config, train=False),
            collate_fn=model_gen.get_collate_fn(task_specs, hparams),
            band_names=config["dataset"]["band_names"],
            format=config["dataset"]["format"],
        )

        ckpt_path = config["model"].get("ckpt_path", None)

        trainer.fit(model, datamodule, ckpt_path=ckpt_path)
        trainer.test(model, datamodule)


def start():
    """Start sweeping."""
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="sweep_trainer.py",
        description="Trains the model using job information contained in the current directory.",
    )
    parser.add_argument("--job-dir", help="Path to the job.", required=True)

    args = parser.parse_args()

    train(args.job_dir)


if __name__ == "__main__":
    start()
