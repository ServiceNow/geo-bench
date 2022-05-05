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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ray import tune
import wandb
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from ray.tune.integration.wandb import WandbLoggerCallback


def train(config, model_gen, job_dir) -> None:
    """Train function for Model Generator and Data Module.
    
    Args:
        config: ray tune config contains hyperparameters being tuned with Ray
        model_gen: model generator
        job_dir: job directory
    """
    job = Job(job_dir)
    hparams = job.hparams
    
    # overwrite job hparams with ray tune config so that they are being tuned
    for h_param, val in config.items():
        hparams[h_param] = val
    
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
    )
    # using wand b logger creates an additional init that shows up in the logs but does not 
    # do anything
    logger_type = "csv"
    loggers = [pl.loggers.CSVLogger(str(job.dir), name="lightning_logs")]
    if logger_type is None:
        logger_type = ""
    if logger_type.lower() == "wandb":
        loggers.append(
            pl.loggers.WandbLogger(project="ccb", name=hparams.get("name", str(job.dir)), save_dir=str(job.dir))
        )
    elif logger_type.lower() == "csv":
        pass  # csv in in loggers by default
    else:
        raise ValueError(f"Logger type ({logger_type}) not recognized.")

    trainer = pl.Trainer(
        gpus=hparams.get("n_gpus", 1),
        max_epochs=hparams["max_epochs"],
        max_steps=hparams.get("train_iters", None),
        limit_val_batches=hparams.get("limit_val_batches", 1.0),
        limit_test_batches=hparams.get("limit_val_batches", 1.0),
        val_check_interval=hparams.get("val_check_interval", 1.0),
        accelerator=hparams.get("accelerator", None),
        deterministic=hparams.get("deterministic", False),
        progress_bar_refresh_rate=0,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=hparams.get("patience", 100)),
            TuneReportCallback( # add this callback to report metrics to RayTune, find a way to pull them from the model generator
                    {
                        "loss": "val_loss",
                    },
                    on="validation_end"
            )
            ],
        logger=loggers,
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
    
# function that handles the hyperparamet tuning
def tune_train_with_ray(job_dir, model_gen) -> None:
    """Tune training function with Asynchroneous Hyperband Scheduler.
    
    Args:
        job_dir: job directory for train function
        model_gen: model generator for train function
    """
    job = Job(job_dir)
    config = job.hparams
    ray_config = job.hparams_ray["ray_params"]

    # add all standard job.hparams to ray_config so that they are stated in wandb
    # but do not over write tunable parameters
    for h_param, val in config.items():
        if h_param not in ray_config:
            ray_config[h_param] = val
    
    scheduler = ASHAScheduler(
        max_t=config["max_epochs"],
        grace_period=1,
        reduction_factor=2)

    # output to log.out where hyperparam tuning progess is printed
    reporter = CLIReporter(
        parameter_columns = list(ray_config.keys()), # needs to match config tunable hyperparameters
        metric_columns=["loss", "training_iteration"]) # need to match metrics to track

    # wrapper in order to call the train function with its specific arguments
    # specify the train function and constant arguments separately after
    # config will change, so only specify constants that will stay the same across
    # all hyperparameter search trials
    train_fn_with_parameters = tune.with_parameters(train, 
                                                    model_gen=model_gen,
                                                    job_dir=job_dir)

    resources_per_trial = {"cpu": config["cpus_per_trial"], "gpu": config["gpus_per_trial"]}

    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    analysis = tune.run(train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="loss",
        mode="min",
        config=ray_config, # this config will be passed to train function and changes
        num_samples=config["num_hyp_samples"],
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_with_ray", # name of directory in job's dataset directory where ray results are saved
        callbacks=[
            WandbLoggerCallback(api_key=os.environ.get("WANDB_API_KEY"), project="ccb")
        ],
        local_dir=job_dir # save ray output in dataset job dir
        )

    print("Best hyperparameters found were: ", analysis.best_config)


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

    tune_train_with_ray(
        model_gen=get_model_generator(args.model_generator), 
        job_dir=args.job_dir,
    )


if __name__ == "__main__":
    start()
