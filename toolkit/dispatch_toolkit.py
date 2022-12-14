import argparse
import subprocess
from pathlib import Path
from subprocess import PIPE

from ruamel.yaml import YAML


# XXX: This needs to appear before the constants since it is used to extract user toolkit info
def _run_shell_cmd(cmd: str, hide_stderr=False):
    """Run a shell command and return the output"""
    result = subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE if hide_stderr else None, universal_newlines=True)
    return result.stdout.replace("\n", "")


# Toolkit user identity (extracted automatically)
TOOLKIT_USER_ORG = _run_shell_cmd("eai organization get --field name")
TOOLKIT_USER_ACCOUNT = _run_shell_cmd("eai account get --field name")
TOOLKIT_USER_ACCOUNT = "snow.rg_climate_benchmark.nilslehmann"
TOOLKIT_USER = f"{TOOLKIT_USER_ORG}.{TOOLKIT_USER_ACCOUNT}"

# Get current git branch (used to tag code by user+branch)
GIT_BRANCH = _run_shell_cmd("git rev-parse --abbrev-ref HEAD")

# Infrastructure setup variables
# Note: Give access to new users by adding them to the eai.rg_climate_benchmark.admin role.
TOOLKIT_IMAGE = "registry.console.elementai.com/snow.rg_climate_benchmark/base:main"
TOOLKIT_DATA = "snow.rg_climate_benchmark.data"
TOOLKIT_CODE = "snow.rg_climate_benchmark.code"
TOOLKIT_CODE_VERSION = f"{TOOLKIT_USER}_{GIT_BRANCH}"
TOOLKIT_BOOTSTRAP_CMD = 'cp -r /mnt/code/* /tmp && cd /tmp && poetry build && pip install dist/climate_change_benchmark-*.whl && export PATH=$PATH:/tmp/.local/bin && echo "Bootstrap completed. Starting execution...\\n\\n\\n"'
TOOLKIT_ENVS = (
    "CC_BENCHMARK_SOURCE_DATASETS=/mnt/data/cc_benchmark/source",
    "CC_BENCHMARK_CONVERTED_DATASETS=/mnt/data/cc_benchmark/converted",
)

# Computational requirements
TOOLKIT_CPU = 4
TOOLKIT_GPU = 1
TOOLKIT_MEM = 32


def _load_envs():
    """Load environment variables that must be set in toolkit"""
    return [env.strip() for env in open(".envs", "r")]


def toolkit_job(script_path: Path, env_vars=()):
    """Launch a job on toolkit along with a specific script (assumed runnable with sh).

    Args:
        script_path: path to .sh script to exectute on toolkit
    """
    job_name = (
        script_path.parent.name.lower()
    )  # TODO actually job_name needs to be unique across all jobs? maybe we need to rethink that.
    for char in (".", "="):  # TODO replace all non alpha numeric chars by '_'.
        job_name = job_name.replace(char, "_")

    # General job config
    cmd = f"eai job new -i {TOOLKIT_IMAGE} --restartable".split(" ")

    # Set job name
    # cmd += ["--name", job_name]  # TODO: Job names cause issues and are not super useful

    # Computational requirements
    cmd += ["--cpu", str(TOOLKIT_CPU)]
    cmd += ["--gpu", str(TOOLKIT_GPU)]
    cmd += ["--mem", str(TOOLKIT_MEM)]
    cmd += ["--account", str(TOOLKIT_USER_ACCOUNT)]

    cmd += ["--gpu-model-filter", "!A100"]
    # cmd += ["--gpu-model-filter", "v100-sxm2-32gb"]

    # Mount data objects
    cmd += ["--data", f"{TOOLKIT_DATA}:/mnt/data"]
    cmd += ["--data", f"{TOOLKIT_CODE}@{TOOLKIT_CODE_VERSION}:/mnt/code"]

    # Set all environment variables
    for e in TOOLKIT_ENVS + env_vars:
        cmd += ["--env", e]

    # TODO: faire poetry install on boot
    cmd += [
        "--",
        f"sh -c '{TOOLKIT_BOOTSTRAP_CMD} && cd /mnt/code/ && sh {str(script_path)}'",
    ]  # TODO do we have to put absolute path? Could we use relative path of script?

    # Launch the job
    print(" ".join(cmd))
    _run_shell_cmd(" ".join(cmd))
    print("Launched.")


def toolkit_dispatcher(exp_dir, prompt=True, env_vars=()) -> None:
    """Scans the exp_dir for experiments to launch"""
    exp_dir = Path(exp_dir)

    print(f"Scanning {exp_dir}.")
    # this is how standard/old version expirements are found
    script_list = list(exp_dir.glob("**/run.sh"))
    if script_list:
        if prompt:
            ds_scripts = list(set([str(p).split("/")[-3] for p in script_list]))
            print("Will launch all of these jobs on toolkit:")
            for ds in ds_scripts:
                print(ds)
                ds_scripts = [script for script in script_list if ds in str(script)]
                for s in ds_scripts:
                    print(str(s))
                ans = input("Ready to proceed? y/n.")
                if ans != "y":
                    continue
                else:
                    for script_path in ds_scripts:
                        toolkit_job(script_path, env_vars)
        return

    # script list for seeded_runs
    script_list = list(exp_dir.glob("**/**/run.sh"))
    if script_list:
        return

    config_list = list(exp_dir.glob("**/**/config.yaml"))
    if config_list:
        # need to decide between seeded runs and sweeps
        config_list = list(exp_dir.glob("**/**/config.yaml"))

        for config_path in config_list:
            yaml = YAML()
            with open(config_path, "r") as yamlfile:
                config = yaml.load(yamlfile)

            model_name = config["model"]["model_name"]

            # ans = input(f"Launch, {model_name} on {config_path.parents[0].name} y/n.")
            # if ans != "y":
            #     continue

            assert "sweep_config_path" in config["wandb"]["sweep"]

            sweep_name = config_path.parents[1].name + "_" + config_path.parents[0].name + "_" + model_name
            sweep_path = config_path.parents[0] / "sweep_config.yaml"

            cmd = ["wandb", "sweep", "--name", sweep_name, str(config_path.parents[0] / "sweep_config.yaml")]

            result = subprocess.run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
            output = result.stderr.replace("\n", "")  # wandb sweep over stderr not stdout unexpectedly

            # need to find a proper way to match with regex
            # sweep_regex = re.compile(r"""
            #     (?P<sweep_phrase>[Run sweep agent with: ])
            # """, re.VERBOSE)

            # match = re.search(sweep_regex, output)
            # if "sweep_id" in match.groupdict():
            #     sweep_id = match.group("sweep_id")
            if "Run sweep agent with: " in output:
                wandb_agent_command = output.split("Run sweep agent with: ")[-1]
                sweep_id = wandb_agent_command.split(" ")[-1]
                config["wandb"]["sweep"]["sweep_id"] = sweep_id
                config["wandb"]["wandb_group"] = sweep_name
                config["wandb"]["sweep"]["sweep_config_path"] = str(sweep_path)
            else:
                raise ValueError(f"Sweep could not be launched successfully, got {output}")

            with open(config_path.parents[0] / "config.yaml", "w") as yamlfile:
                yaml.dump(config, yamlfile)

            num_agents = config["wandb"]["sweep"]["num_agents"]
            num_trials = config["wandb"]["sweep"]["num_trials_per_agent"]

            # save a script that can be launched with toolkit
            script_path = config_path.parents[0] / "run.sh"
            with open(script_path, "w") as fd:
                fd.write("#!/bin/bash\n")
                fd.write(f"{wandb_agent_command} --count {num_trials}")

            # launch agents via toolkit
            for agent in range(num_agents):
                toolkit_job(script_path, env_vars)


def push_code(dir):
    """Push the local code to the cluster"""
    print("Pushing code...")
    dir = ".."
    _run_shell_cmd(f"eai data branch add {TOOLKIT_CODE}@empty {TOOLKIT_CODE_VERSION}", hide_stderr=True)
    _run_shell_cmd(f"eai data content rm {TOOLKIT_CODE}@{TOOLKIT_CODE_VERSION} .", hide_stderr=False)

    _run_shell_cmd(f"rsync -a '{dir}/' /tmp/rg_climate_benchmark --delete --exclude-from='{dir}/toolkit/.eaiignore'")
    _run_shell_cmd(f"eai data push {TOOLKIT_CODE}@{TOOLKIT_CODE_VERSION} /tmp/rg_climate_benchmark:/")
    _run_shell_cmd("rm -rf /tmp/rg_climate_benchmark")


def start():
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="sequential_dispatcher.py",
        description="Sequentially dispatch all run.sh in the experiment directory.",
    )

    parser.add_argument(
        "--experiment-dir",
        help="The base directory in which experiment-related files should be created.",
        required=True,
    )

    parser.add_argument(
        "--code-dir",
        help="The directory that contains the ccb package (default='.').",
        default=".",
    )

    args = parser.parse_args()
    push_code(args.code_dir)
    toolkit_dispatcher(args.experiment_dir)


if __name__ == "__main__":
    start()
