#!/usr/bin/env python
"""
Trains the model using job information contained in the current directory.
Intented to be used for wandb sweeps where agents execute training
with a given set of hyperparameters.
Expects to find files "hparams.json" and "task_specs.json".
Usage: sweep-trainer.py --model-generator path/to/my/model/generator.py --job-dir path/to/job/dir
"""
import argparse

from ccb.torch_toolbox.dataset import DataModule
from ccb.experiment.experiment import get_model_generator, Job
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import wandb


def train(model_gen, job_dir) -> None:
    """Train a model from the model generator on datamodule.

    Args:
        model_gen: model generator
        job_dir: job directory that contains task_specs and hparams.json

    """
    job = Job(job_dir)
    hparams = job.hparams
    seed = hparams.get("seed", None)
    if seed is not None:
        pl.seed_everything(seed, workers=True)

    with wandb.init(dir=job_dir, config=hparams, group=hparams["wandb_group"], allow_val_change=True) as run:

        wandb_config = run.config
        # set up W&B logger
        wandb_logger = pl.loggers.WandbLogger(
            project="ccb",
            entity="climate-benchmark",
            id=None,
            group=wandb_config.get("wandb_group", None),
            name=wandb_config.get("name", None),
            save_dir=str(job.dir),
            resume=True,
        )

        loggers = [pl.loggers.CSVLogger(str(job.dir), name="csv_logs"), wandb_logger]

        print(wandb.config)
        # instantiate model - need to used wandb config hparams here that the sweep overwrites
        model = model_gen.generate(task_specs=job.task_specs, hyperparameters=wandb.config)

        datamodule = DataModule(
            job.task_specs,
            batch_size=wandb_config["batch_size"],
            num_workers=wandb_config["num_workers"],
            train_transform=model_gen.get_transform(job.task_specs, wandb_config, train=True),
            eval_transform=model_gen.get_transform(job.task_specs, wandb_config, train=False),
            collate_fn=model_gen.get_collate_fn(job.task_specs, wandb_config),
            band_names=wandb_config.get("band_names", ("red", "green", "blue")),
            format=wandb_config.get("format", "hdf5"),
        )

        ckpt_dir = os.path.join(job_dir, "checkpoint")
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            every_n_epochs=1,
        )

        trainer = pl.Trainer(
            default_root_dir=job_dir,
            gpus=wandb_config.get("n_gpus", 1),
            max_epochs=wandb_config["max_epochs"],
            max_steps=wandb_config.get("train_iters", -1),
            limit_val_batches=wandb_config.get("limit_val_batches", 1.0),
            limit_test_batches=wandb_config.get("limit_val_batches", 1.0),
            val_check_interval=wandb_config.get("val_check_interval", 1.0),
            accelerator=wandb_config.get("accelerator", None),
            deterministic=wandb_config.get("deterministic", False),
            log_every_n_steps=wandb_config.get("log_every_n_steps", 10),
            enable_progress_bar=wandb_config.get("enable_progress_bar", False),
            fast_dev_run=wandb_config.get("fast_dev_run", False),
            accumulate_grad_batches=wandb_config.get("accumulate_grad_batches", 1),
            callbacks=[
                EarlyStopping(
                    monitor="val_loss", mode="min", patience=wandb_config.get("patience", 20), min_delta=1e-5
                ),
                checkpoint_callback,
            ],
            logger=loggers,
            precision=16,
        )

        ckpt_path = wandb_config.get("ckpt_path", None)

        trainer.fit(model, datamodule, ckpt_path=ckpt_path)
        trainer.test(model, datamodule)


def start():
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="sweep_trainer.py",
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
