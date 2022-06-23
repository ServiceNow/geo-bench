#!/usr/bin/env python
"""Train the model using job information contained in the current directory."""

import argparse
import os
import random
import string

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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

    model = model_gen.generate(task_specs=job.task_specs, hparams=hparams, config=config)
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

    run_id = "".join(random.SystemRandom().choice(string.ascii_lowercase + string.digits) for _ in range(8))
    config["wandb"]["wandb_run_id"] = run_id
    loggers = [
        pl.loggers.CSVLogger(str(job.dir), name="lightning_logs"),
        pl.loggers.WandbLogger(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            id=run_id,
            group=config["wandb"].get("wandb_group", None),
            name=config["wandb"].get("name", None),
            save_dir=str(job.dir),
            resume="allow",
            config=hparams,
        ),
    ]
    job.save_config(config, overwrite=True)

    ckpt_dir = os.path.join(job_dir, "checkpoint")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir, save_top_k=1, monitor="val_loss", mode="min", every_n_epochs=1
    )

    trainer = pl.Trainer(
        **config["pl"],
        default_root_dir=job_dir,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=hparams.get("patience", 30), min_delta=1e-5),
            checkpoint_callback,
        ],
        logger=loggers,
    )

    ckpt_path = config["model"].get("ckpt_path", None)

    trainer.fit(model, datamodule, ckpt_path=ckpt_path)
    trainer.test(model, datamodule)


def start() -> None:
    """Start training."""
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="trainer.py", description="Trains the model using job information contained in the current directory."
    )

    parser.add_argument("--job_dir", help="Path to the job.", required=True)

    args = parser.parse_args()

    train(args.job_dir)


if __name__ == "__main__":
    start()
