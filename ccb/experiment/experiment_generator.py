"""
Generate experiment directory structure
Usage: experiment_generator.py --model-generator path/to/my/model/generator.py  --experiment-dir path/to/my/experiments
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from ccb import io
from ccb.experiment.experiment import Job, get_model_generator


def experiment_generator(
    model_generator_module_name: str,
    experiment_dir: str,
    task_filter: callable = None,
    benchmark_name: str = "default",
    experiment_name: str = None,
    experiment_type: str = "standard",
):
    """
    Generates the directory structure for every tasks and every hyperparameter configuration.
    According to model_generator.hp_search.
    Parameters:
    -----------
    model_generator: ModelGenerator
        The generator associated with the current model. Used to get hyperparameter combinations.
    experiment_dir: str
        The directory in which to create the experiment directories.
    task_filter: callable(TaskSpecification)
        A function that takes as input a task specification instance and returns False if it should be skipped.
    benchmark_name: str
        The name of the benchmark on which to conduct the experiment (default: "default").
    experiment_name: str
        The name of the current experiment. Will be used as a prefix to the results directory (default: None).
    experiment_type: what kind of experiment to dispatch, ["sweep", "seeded_runs", "standard"]

    Returns:
        Name of the experiment.
    """
    experiment_dir = Path(experiment_dir)
    experiment_prefix = (
        f"{experiment_name or 'experiment'}_{benchmark_name}_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"
    )
    if experiment_name is not None:
        experiment_dir /= experiment_prefix

    print(f"Generating experiments for {model_generator_module_name} on {benchmark_name} benchmark.")

    for task_specs in io.task_iterator(benchmark_name=benchmark_name):
        if task_filter is not None:
            if not task_filter(task_specs):
                continue

        print(task_specs.dataset_name)

        if experiment_type == "sweep":
            model_generator = get_model_generator(model_generator_module_name)

            base_hparams = model_generator.base_hparams

            # use wandb sweep for hyperparameter search
            model = model_generator.generate(task_specs, base_hparams)
            hparams = model.hyperparameters

            hparams["dataset_name"] = task_specs.dataset_name
            hparams["benchmark_name"] = benchmark_name
            hparams["model_generator_name"] = model_generator_module_name

            # create and fill experiment directory
            job_dir = experiment_dir / task_specs.dataset_name
            job = Job(job_dir)
            job.save_hparams(hparams)
            job.save_task_specs(task_specs)

            job.write_wandb_sweep_cl_script(
                model_generator_module_name, job_dir=job_dir, base_sweep_config=hparams["sweep_config_yaml_path"]
            )

        elif experiment_type == "seeded_runs":
            NUM_SEEDS = 3

            # use wandb sweep for hyperparameter search
            with open(
                "/mnt/home/climate-change-benchmark/analysis_tool/results/seeded_runs_best_hparams.json", "r"
            ) as f:
                best_params = json.load(f)

            backbone_names = list(best_params[benchmark_name][task_specs.dataset_name].keys())

            for back_name in backbone_names:
                backbone_config = best_params[benchmark_name][task_specs.dataset_name][back_name]

                model_generator_name = backbone_config["model_generator_name"]

                model_generator = get_model_generator(model_generator_name, hparams=backbone_config)

                backbone_config["wandb_group"] = task_specs.dataset_name + "/" + back_name + "/" + experiment_prefix
                backbone_config["benchmark_name"] = benchmark_name

                for i in range(NUM_SEEDS):
                    # set seed to be used in experiment
                    backbone_config["seed"] = i

                    job_dir = experiment_dir / task_specs.dataset_name / back_name / f"seed_{i}"
                    job = Job(job_dir)
                    job.save_hparams(backbone_config)
                    job.save_task_specs(task_specs)
                    job.write_script(model_generator_name, job_dir=job_dir)

        else:
            # single run of a model
            model_generator = get_model_generator(model_generator_module_name)

            base_hparams = model_generator.base_hparams

            # use wandb sweep for hyperparameter search
            model = model_generator.generate(task_specs, base_hparams)
            hparams = model.hyperparameters

            hparams["dataset_name"] = task_specs.dataset_name
            hparams["benchmark_name"] = benchmark_name
            hparams["model_generator_name"] = model_generator_module_name
            if model_generator_module_name != "ccb.torch_toolbox.model_generators.py_segmentation_generator":
                hparams["name"] = f"{experiment_prefix}/{task_specs.dataset_name}/{hparams['backbone']}"
            else:
                hparams[
                    "name"
                ] = f"{experiment_prefix}/{task_specs.dataset_name}/{hparams['encoder_type']}/{hparams['decoder_type']}"

            # create and fill experiment directory
            job_dir = experiment_dir / task_specs.dataset_name
            print("job dir")
            print(job_dir)
            job = Job(job_dir)
            job.save_hparams(hparams)
            job.save_task_specs(task_specs)

            job.write_script(model_generator_module_name, job_dir=job_dir)

    return experiment_dir


def start():
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="experiment_generator.py",
        description="Generate experiment directory structure based on user-defined model generator",
    )
    parser.add_argument(
        "--model-generator",
        help="Path to a Python file that defines a model generator (expects a model_generator variable to exist).",
        required=True,
    )
    parser.add_argument(
        "--experiment-dir",
        help="The based directory in which experiment-related files should be created.",
        required=True,
    )

    parser.add_argument(
        "--benchmark",
        help="The set of dataset that will be used for evaluating. 'ccb' | 'mnist' ",
        required=False,
        default="default",
    )

    parser.add_argument(
        "--experiment-name",
        help="An optional name to give to the experiment. Will be used as a prefix to the results directory.",
        required=False,
        default=None,
    )

    args = parser.parse_args()

    experiment_generator(
        args.model_generator, args.experiment_dir, benchmark_name=args.benchmark, experiment_name=args.experiment_name
    )


if __name__ == "__main__":
    start()
