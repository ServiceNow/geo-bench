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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import string
import random
import json


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

    model = model_gen.generate(job.task_specs, hparams)

    datamodule = DataModule(
        job.task_specs,
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        train_transform=model_gen.get_transform(job.task_specs, hparams, train=True),
        eval_transform=model_gen.get_transform(job.task_specs, hparams, train=False),
        collate_fn=model_gen.get_collate_fn(job.task_specs, hparams),
        band_names=hparams.get("band_names", ("red", "green", "blue")),
        format=hparams.get("format", "hdf5"),
    )

    logger_type = hparams.get("logger", None)
    loggers = [pl.loggers.CSVLogger(str(job.dir), name="lightning_logs")]
    if logger_type is None:
        logger_type = ""
    if logger_type.lower() == "wandb":
        run_id = "".join(random.SystemRandom().choice(string.ascii_lowercase + string.digits) for _ in range(8))
        hparams["wandb_run_id"] = run_id
        with open(os.path.join(job_dir, "hparams.json"), "w") as f:
            json.dump(hparams, f)

        loggers.append(
            pl.loggers.WandbLogger(
                project="ccb",
                entity="climate-benchmark",
                id=run_id,
                group=hparams.get("wandb_group", None),
                name=hparams.get("name", None),
                save_dir=str(job.dir),
                resume="allow",
                config=hparams,
            )
        )

    elif logger_type.lower() == "csv":
        pass  # csv in in loggers by default
    else:
        raise ValueError(f"Logger type ({logger_type}) not recognized.")

    ckpt_dir = os.path.join(job_dir, "checkpoint")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir, save_top_k=1, monitor="val_loss", mode="min", every_n_epochs=1
    )

    trainer = pl.Trainer(
        default_root_dir=job_dir,
        gpus=hparams.get("n_gpus", 1),
        max_epochs=hparams["max_epochs"],
        max_steps=hparams.get("train_iters", -1),
        limit_val_batches=hparams.get("limit_val_batches", 1.0),
        limit_test_batches=hparams.get("limit_val_batches", 1.0),
        val_check_interval=hparams.get("val_check_interval", 1.0),
        accelerator=hparams.get("accelerator", None),
        deterministic=hparams.get("deterministic", False),
        log_every_n_steps=hparams.get("log_every_n_steps", 10),
        enable_progress_bar=hparams.get("enable_progress_bar", False),
        fast_dev_run=hparams.get("fast_dev_run", False),
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=hparams.get("patience", 30), min_delta=1e-5),
            checkpoint_callback,
        ],
        logger=loggers,
    )

    ckpt_path = hparams.get("ckpt_path", None)

    trainer.fit(model, datamodule, ckpt_path=ckpt_path)
    trainer.test(model, datamodule)


def start():
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="trainer.py", description="Trains the model using job information contained in the current directory."
    )
    parser.add_argument(
        "--model-generator",
        help="Module name that defines a model generator. Must be in PYTHONPATH and expects a model_generator variable to exist.",
        required=True,
    )
    parser.add_argument("--job-dir", help="Path to the job.", required=True)

    args = parser.parse_args()

    # Load the user-specified model generator
    model_gen = get_model_generator(args.model_generator)
    train(model_gen, args.job_dir)


if __name__ == "__main__":
    start()
