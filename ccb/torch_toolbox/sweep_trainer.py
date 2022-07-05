"""Wandb sweep the model using job information contained in the current directory."""

import argparse
import os

import pytorch_lightning as pl
import wandb
from ruamel.yaml import YAML

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

        wandb_config = run.config  # wandb config now includes all variables that have been changed by sweep
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
        csv_logger = pl.loggers.CSVLogger(str(job.dir), name="csv_logs")

        loggers = [csv_logger, wandb_logger]

        # instantiate model - since model generator relies both on hparams and wandb_config values
        # set for the specific sweep, need to update hparams with wandb valus
        hparams.update(wandb_config)

        model = model_gen.generate_model(task_specs=job.task_specs, hparams=hparams, config=config)

        trainer = model_gen.generate_trainer(config=config, hparams=hparams, job=job)

        # reload config for updates that happened during generation
        config = job.config

        datamodule = DataModule(
            task_specs,
            benchmark_dir=config["experiment"]["benchmark_dir"],
            partition_name=config["experiment"]["partition_name"],
            batch_size=hparams["batch_size"],
            num_workers=config["dataloader"]["num_workers"],
            train_transform=model_gen.get_transform(task_specs=task_specs, hparams=hparams, config=config, train=True),
            eval_transform=model_gen.get_transform(task_specs=task_specs, hparams=hparams, config=config, train=False),
            collate_fn=model_gen.get_collate_fn(task_specs, hparams),
            band_names=config["dataset"]["band_names"],
            format=config["dataset"]["format"],
        )

        # update trainer module
        trainer.loggers = loggers
        trainer.log_every_n_steps = min(len(datamodule.train_dataloader()), config["pl"]["log_every_n_steps"])

        ckpt_path = config["model"].get("ckpt_path", None)

        trainer.fit(model, datamodule, ckpt_path=ckpt_path)
        trainer.test(model, datamodule)

        # save updated configs in csv_logger one directories are created
        csv_logger_dir = csv_logger.log_dir
        yaml = YAML()
        with open(os.path.join(csv_logger_dir, "hparams.yaml"), "w") as fd:
            yaml.dump(hparams, fd)

        with open(os.path.join(csv_logger_dir, "config.yaml"), "w") as fd:
            yaml.dump(config, fd)


def start():
    """Start sweeping."""
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="sweep_trainer.py",
        description="Trains the model using job information contained in the current directory.",
    )
    parser.add_argument("--job_dir", help="Path to the job.", required=True)

    args = parser.parse_args()

    train(args.job_dir)


if __name__ == "__main__":
    start()
