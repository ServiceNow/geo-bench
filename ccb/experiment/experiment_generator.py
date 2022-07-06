"""Generate experiment directory structure.

Usage: experiment_generator.py --model-generator path/to/my/model/generator.py  --experiment-dir path/to/my/experiments
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import yaml

from ccb import io
from ccb.experiment.experiment import Job, get_model_generator


def experiment_generator(
    config_filepath: str,
    hparam_filepath: str,
):
    """Generate the directory structure for every tasks.

    According to model_generator.hp_search.

    Args:
        model_generator_module_name: The generator associated with the current model. Used to get hyperparameter combinations.
        experiment_dir: The directory in which to create the experiment directories.
        task_filter: A function that takes as input a task specification instance and returns False if it should be skipped.
        benchmark_name: The name of the benchmark on which to conduct the experiment (default: "default").
        experiment_name: The name of the current experiment. Will be used as a prefix to the results directory (default: None).
        experiment_type: what kind of experiment to dispatch, ["sweep", "seeded_runs", "standard"]

    Returns:
        Name of the experiment directory.

    Raises:
        FileNotFoundError if path to config file or hparam file does not exist
    """
    config_file_path: Path = Path(config_filepath)
    hparam_file_path: Path = Path(hparam_filepath)
    # check that specified paths exists
    if config_file_path.is_file():
        with config_file_path.open() as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file at path {config_file_path} does not exist.")

    if hparam_file_path.is_file():
        with hparam_file_path.open() as f:
            hparams = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file at path {config_file_path} does not exist.")

    benchmark_dir = config["experiment"]["benchmark_dir"]

    experiment_prefix = f"{config['experiment']['experiment_name'] or 'experiment'}_{os.path.basename(benchmark_dir)}_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"
    if config["experiment"]["experiment_name"] is not None:
        experiment_dir = Path(config["experiment"]["generate_experiment_dir"]) / experiment_prefix

    print(
        f"Generating experiments for {config['model']['model_generator_module_name']} on {os.path.basename(benchmark_dir)} benchmark."
    )

    for task_specs in io.task_iterator(benchmark_dir=benchmark_dir):
        print(task_specs.dataset_name)
        experiment_type = config["experiment"]["experiment_type"]
        if experiment_type == "sweep":
            model_generator = get_model_generator(config["model"]["model_generator_module_name"])

            # use wandb sweep for hyperparameter search
            model = model_generator.generate_model(task_specs, hparams, config)

            # there might be other params added during the generate process,
            # continue with hyperparameters from initialized model
            hparams = model.hyperparameters

            # create and fill experiment directory
            job_dir = experiment_dir / task_specs.dataset_name
            job = Job(job_dir)
            job.save_hparams(hparams)
            job.save_config(config)
            job.save_task_specs(task_specs)

            # sweep name that will be seen on wandb
            if (
                config["model"]["model_generator_module_name"]
                != "ccb.torch_toolbox.model_generators.py_segmentation_generator"
            ):
                name = "_".join(str(job_dir).split("/")[-2:]) + "_" + hparams["backbone"]
            else:
                name = "_".join(str(job_dir).split("/")[-2:]) + "_" + hparams["encoder"] + "_" + hparams["decoder"]

            job.write_wandb_sweep_cl_script(
                config["model"]["model_generator_module_name"],
                job_dir=job_dir,
                base_sweep_config=config["wandb"]["sweep"]["sweep_config_path"],
                name=name,
            )

        elif experiment_type == "seeded_runs":
            NUM_SEEDS = 3

            # use wandb sweep for hyperparameter search
            with open(
                "/mnt/home/climate-change-benchmark/analysis_tool/results/seeded_runs_best_hparams.json", "r"
            ) as f:
                best_params = json.load(f)

            benchmark_name = os.path.basename(config["experiment"]["benchmark_dir"])
            backbone_names = list(best_params[benchmark_name][task_specs.dataset_name].keys())

            for back_name in backbone_names:
                backbone_config = best_params[benchmark_name][task_specs.dataset_name][back_name]

                model_generator_name = backbone_config["model_generator_name"]

                backbone_config["wandb_group"] = task_specs.dataset_name + "/" + back_name + "/" + experiment_prefix
                backbone_config["benchmark_name"] = benchmark_name

                for i in range(NUM_SEEDS):
                    # set seed to be used in experiment
                    backbone_config["seed"] = i

                    job_dir = experiment_dir / task_specs.dataset_name / back_name / f"seed_{i}"
                    job = Job(job_dir)
                    job.save_hparams(backbone_config)
                    job.save_config(config)
                    job.save_task_specs(task_specs)
                    job.write_script(model_generator_name, job_dir=job_dir)

        else:
            # single run of a model
            model_generator_module_name = config["model"]["model_generator_module_name"]
            model_generator = get_model_generator(model_generator_module_name)

            # use wandb sweep for hyperparameter search
            model = model_generator.generate_model(task_specs, hparams, config)
            hparams = model.hyperparameters

            config["experiment"]["dataset_name"] = task_specs.dataset_name
            config["experiment"]["benchmark_name"] = os.path.basename(config["experiment"]["benchmark_dir"])

            if model_generator_module_name != "ccb.torch_toolbox.model_generators.py_segmentation_generator":
                config["experiment"]["name"] = f"{experiment_prefix}/{task_specs.dataset_name}/{hparams['backbone']}"
            else:
                config["experiment"][
                    "name"
                ] = f"{experiment_prefix}/{task_specs.dataset_name}/{hparams['encoder_type']}/{hparams['decoder_type']}"

            # create and fill experiment directory
            job_dir = experiment_dir / task_specs.dataset_name
            job = Job(job_dir)
            job.save_hparams(hparams)
            job.save_config(config)
            job.save_task_specs(task_specs)

            job.write_script(model_generator_module_name, job_dir=job_dir)

    return experiment_dir


def start() -> None:
    """Start generating experiments."""
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="experiment_generator.py",
        description="Generate experiment directory structure based on user-defined model generator",
    )
    parser.add_argument(
        "--config_filepath",
        help="The path to the configuration file.",
        required=True,
    )
    parser.add_argument(
        "--hparam_filepath",
        help="The path to model hparam file.",
        required=True,
    )

    args = parser.parse_args()

    experiment_generator(config_filepath=args.config_filepath, hparam_filepath=args.hparam_filepath)


if __name__ == "__main__":
    start()
