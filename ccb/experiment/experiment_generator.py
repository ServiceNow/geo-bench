"""Generate experiment directory structure.

Usage: experiment_generator.py --model-generator path/to/my/model/generator.py  --experiment-dir path/to/my/experiments
"""
import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path

import yaml

from ccb import io
from ccb.experiment.experiment import Job, get_model_generator


def define_model_name(config):
    """Define a model name."""
    if config["model"]["model_generator_module_name"] == "ccb.torch_toolbox.model_generators.ssl_moco":
        model_name = config["model"]["ssl_method"] + "_" + config["model"]["backbone"]
        # model_name = "ssl_moco_" + config["model"]["backbone"]
    elif config["model"]["model_generator_module_name"] == "ccb.torch_toolbox.model_generators.timm_generator":
        if config["model"]["pretrained"]:
            model_name = config["model"]["backbone"]
        else:
            model_name = "scratch_" + config["model"]["backbone"]
    elif config["model"]["model_generator_module_name"] == "ccb.torch_toolbox.model_generators.conv4":
        model_name = config["model"]["backbone"]
    elif config["model"]["model_generator_module_name"] == "ccb.torch_toolbox.model_generators.wang_rs_pretrained":
        model_name = "millionaid_" + config["model"]["backbone"]
    elif config["model"]["model_generator_module_name"] == "ccb.torch_toolbox.model_generators.seco":
        model_name = "seco_" + config["model"]["backbone"]
    elif config["model"]["model_generator_module_name"] == "ccb.torch_toolbox.model_generators.swin_segmentation":
        model_name = config["model"]["model"] + "_segmentation"
    else:
        if config["model"]["pretrained"] is False:
            model_name = "scratch_" + config["model"]["encoder_type"] + "_" + config["model"]["decoder_type"]
        else:
            model_name = config["model"]["encoder_type"] + "_" + config["model"]["decoder_type"]
    return model_name


def experiment_generator(
    config_filepath: str,
) -> Path:
    """Generate the directory structure for every tasks.

    According to model_generator.hp_search.

    Args:
        config_filepath: path to config file that defines experiment
    Returns:
        Name of the experiment directory.

    Raises:
        FileNotFoundError if path to config file or hparam file does not exist
    """
    config_file_path = Path(config_filepath)

    # check that specified paths exists
    if config_file_path.is_file():
        with config_file_path.open() as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file at path {config_file_path} does not exist.")

    benchmark_dir = config["experiment"]["benchmark_dir"]

    model_name = define_model_name(config)

    experiment_prefix = f"{config['experiment']['experiment_name'] or 'experiment'}_{os.path.basename(benchmark_dir)}_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}_{model_name}"
    if config["experiment"]["experiment_name"] is not None:
        experiment_dir: Path = Path(config["experiment"]["generate_experiment_dir"]) / experiment_prefix
    else:
        experiment_dir: Path = Path(config["experiment"]["generate_experiment_dir"])  # type: ignore[no-redef]

    # find the batch size for the model/dataset combination
    with open(str(Path(__file__).parent.parent / "torch_toolbox/wandb/batch_sizes.json"), "r") as f:
        batch_size_dict = json.load(f)

    for task_specs in io.task_iterator(benchmark_dir=benchmark_dir):
        print(task_specs.dataset_name)
        experiment_type = config["experiment"]["experiment_type"]
        task_config = copy.deepcopy(config)

        if experiment_type == "sweep":

            beyond_rgb = False
            if task_config["dataset"]["band_names"] == "all":
                if task_specs.dataset_name not in ["eurosat", "brick_kiln_v1.0", "bigearthnet", "so2sat"]:
                    continue
                band_names = [band_info.name for band_info in task_specs.bands_info]
                beyond_rgb = True
                if "Cloud Probability" in band_names:
                    band_names.remove("Cloud Probability")
                if task_specs.dataset_name == "so2sat":  # only sentinel 2 bands for so2sat
                    band_names = [band_info for band_info in band_names if "VH." not in band_info]
                    band_names = [band_info for band_info in band_names if "VV." not in band_info]

                task_config["dataset"]["band_names"] = band_names

            print(len(task_config["dataset"]["band_names"]))
            model_generator = get_model_generator(config["model"]["model_generator_module_name"])

            # use wandb sweep for hyperparameter search
            model = model_generator.generate_model(task_specs, task_config)

            # there might be other params added during the generate process,
            # continue with hyperparameters from initialized model
            task_config = model.config

            task_config["model"]["model_name"] = model_name

            task_config["model"]["batch_size"] = batch_size_dict[model_name][task_specs.dataset_name]

            if beyond_rgb:
                task_config["model"]["batch_size"] = int(task_config["model"]["batch_size"] / 2)

            # create and fill experiment directory
            job_dir = experiment_dir / task_specs.dataset_name
            job = Job(job_dir)
            job.save_config(task_config)
            job.save_task_specs(task_specs)

            # sweep name that will be seen on wandb
            wandb_name = "_".join(str(job_dir).split("/")[-2:]) + "_" + model_name

            job.write_wandb_sweep_cl_script(
                task_config["model"]["model_generator_module_name"],
                job_dir=job_dir,
                base_sweep_config=task_config["wandb"]["sweep"]["sweep_config_path"],
                name=wandb_name,
            )

        elif experiment_type == "seeded_runs":

            with open(config["experiment"]["best_hparam_config"], "r") as f:
                seed_run_dict = json.load(f)

            part_name = task_config["experiment"]["partition_name"].split("_partition.json")[0]

            ds_dict = seed_run_dict[model_name][part_name]

            # for datasets that are in classification benchmark dir but not swept yet
            try:
                exp_dir = ds_dict[task_specs.dataset_name]
            except KeyError:
                continue
            # load best config file
            with open(os.path.join(exp_dir, "config.yaml")) as f:
                best_config = yaml.safe_load(f)

            best_config["wandb"]["wandb_group"] = task_specs.dataset_name + "/" + model_name + "/" + experiment_prefix

            print(best_config["model"]["batch_size"])
            print(exp_dir)

            for i in range(config["experiment"]["num_seeds"]):
                # set seed to be used in experiment
                best_config["experiment"]["seed"] = i

                job_dir = experiment_dir / task_specs.dataset_name / f"seed_{i}"
                job = Job(job_dir)
                job.save_config(best_config)
                job.save_task_specs(task_specs)
                job.write_script(job_dir=str(job_dir))

        else:
            # single run of a model
            model_generator_module_name = config["model"]["model_generator_module_name"]
            model_generator = get_model_generator(model_generator_module_name)

            model = model_generator.generate_model(task_specs, config)
            config = model.config

            config["experiment"]["dataset_name"] = task_specs.dataset_name
            config["experiment"]["benchmark_name"] = os.path.basename(config["experiment"]["benchmark_dir"])

            if model_generator_module_name != "ccb.torch_toolbox.model_generators.py_segmentation_generator":
                config["experiment"][
                    "name"
                ] = f"{experiment_prefix}/{task_specs.dataset_name}/{config['model']['backbone']}"
            else:
                config["experiment"][
                    "name"
                ] = f"{experiment_prefix}/{task_specs.dataset_name}/{config['model']['encoder_type']}/{config['model']['decoder_type']}"

            # create and fill experiment directory
            job_dir = experiment_dir / task_specs.dataset_name
            job = Job(job_dir)
            job.save_config(config)
            job.save_task_specs(task_specs)
            print(job_dir)
            job.write_script(job_dir=str(job_dir))

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

    args = parser.parse_args()

    experiment_generator(config_filepath=args.config_filepath)


if __name__ == "__main__":
    start()
