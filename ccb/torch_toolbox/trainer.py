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

from ray import tune
import wandb
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback

from ray.tune.integration.wandb import (
    WandbLoggerCallback,
    WandbTrainableMixin,
    wandb_mixin,
)


def train(config, model_gen, job_dir, num_gpus=1, num_epochs=10) -> None:
    """Train function for Model Generator and Data Module.
    
    Args:
        config: config contains non-constant hyperparameters being tuned with Ray
        model_gen: model generator
        job_dir: job directory
        num_gpus: number of gpus to use for trainer
        num_epochs: number of epochs

    """
    job = Job(job_dir)
    hparams = job.hparams
    # hack to set learning rate from config
    # have to change the use of the hparams.json file
    hparams["lr_head"] = config["lr_head"]
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
        gpus=num_gpus, #, was # gpus=hparams.get("n_gpus", 1),
        max_epochs = num_epochs, # constants, was max_epochs=hparams["max_epochs"],
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
def tune_train_asha(job_dir, model_gen, your_wandb_api_key, num_samples=3, num_epochs=10, gpus_per_trial=1) -> None:
    """Tune training function with Asynchroneous Hyperband Scheduler.
    
    Args:
        job_dir: job directory for train function
        model_gen: model generator for train function
        your_wandb_api_key: wandb api key
        num_samples: number of samples for hyperparameters
        num_epochs: max num epochs for trainer
        gpus_per_trial: per single hyperparameter trial how many gpus to use
    
    """
    # specify all hyperparameters to search that will be used in train function above
    config = {
        "lr_head": tune.loguniform(1e-4, 1e-1),
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns = list(config.keys()), # needs to match config tunable hyperparameters
        metric_columns=["loss", "training_iteration"]) # need to match metrics to track

    # wrapper in order to call the train function with its specific arguments
    # specify the train function and constant arguments separately after
    # config will change, so only specify constants that will stay the same across
    # all hyperparameter search trials
    train_fn_with_parameters = tune.with_parameters(train, 
                                                    model_gen=model_gen,
                                                    job_dir=job_dir,
                                                    num_epochs=num_epochs,
                                                    num_gpus=gpus_per_trial)

    resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}

    wandb.login(key=your_wandb_api_key)

    analysis = tune.run(train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="loss",
        mode="min",
        config=config, # this config will be passed to train function and changes
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_train_asha",
        callbacks=[
            WandbLoggerCallback(api_key=your_wandb_api_key, project="CCB")
        ],
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
    api_key = "def8d0fad10d1479d79ab4c7e68530d59be04cf5"
    # Load the user-specified model generator
    model_gen = get_model_generator(args.model_generator)
    # train(model_gen, args.job_dir)
    tune_train_asha(
        model_gen=model_gen, 
        job_dir=args.job_dir,
        your_wandb_api_key = api_key
    )



if __name__ == "__main__":
    start()
