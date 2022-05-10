"""
Generate experiment directory structure

Usage: experiment_generator.py --model-generator path/to/my/model/generator.py  --experiment-dir path/to/my/experiments

"""
import argparse

from datetime import datetime
from pathlib import Path

from ccb.experiment.experiment import Job
from ccb.experiment.experiment import get_model_generator
from ccb import io


def experiment_generator(
    model_generator_module_name: str,
    experiment_dir: str,
    task_filter: callable = None,
    max_num_configs: int = 10,
    benchmark_name: str = "default",
    experiment_name: str = None,
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

    Returns:
        Name of the experiment.
    """
    experiment_dir = Path(experiment_dir)
    experiment_prefix = f"{experiment_name or 'experiment'}_{benchmark_name}_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"
    if experiment_name is not None:
        experiment_dir /= experiment_prefix

    model_generator = get_model_generator(model_generator_module_name)

    print(f"Generating experiments for {model_generator_module_name} on {benchmark_name} benchmark.")
    
    for task_specs in io.task_iterator(benchmark_name=benchmark_name):
        if task_filter is not None:
            if not task_filter(task_specs):
                continue
        print(task_specs.dataset_name)

        if "use_sweep_cl" in model_generator.base_hparams and model_generator.base_hparams["use_sweep_cl"] is True:
            #use wandb sweep for hyperparameter search
            
            # # add sweep id to parameters to run script later
            hparams = model_generator.base_hparams
            hparams["name"] = f"{experiment_prefix}/{task_specs.dataset_name}/{hparams['backbone']}"

            # create and fill experiment directory
            job_dir = experiment_dir / task_specs.dataset_name
            job = Job(job_dir)
            job.save_hparams(hparams)
            job.save_task_specs(task_specs)

            job.write_wandb_sweep_cl_script(
                model_generator_module_name, 
                job_dir=job_dir,
                base_sweep_config=hparams["sweep_config_yaml_path"],
            )

            continue
        
        for hparams, hparams_string in model_generator.hp_search(task_specs, max_num_configs):

            # Override hparams["name"] parameter in hparams - forwarded to wandb in trainer.py
            hparams['name'] = f'{experiment_prefix}/{task_specs.dataset_name}'#/{hparams_string}'

            # Create and fill experiment directory
            job_dir = experiment_dir / task_specs.dataset_name / hparams_string
            job = Job(job_dir)
            print("  ", hparams_string, " -> hparams['name']=", hparams['name'])
            job.save_hparams(hparams)
            job.save_task_specs(task_specs)
            job.write_script(model_generator_module_name, script_name="ccb-trainer", job_dir=job_dir)

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

    # Generate experiments
    experiment_generator(
        args.model_generator, args.experiment_dir, benchmark_name=args.benchmark, experiment_name=args.experiment_name
    )


if __name__ == "__main__":
    start()
