import argparse
from pathlib import Path
import subprocess

from subprocess import PIPE


# XXX: This needs to appear before the constants since it is used to extract user toolkit info
def _run_shell_cmd(cmd: str, hide_stderr=False):
    """Run a shell command and return the output"""
    result = subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE if hide_stderr else None, universal_newlines=True)
    return result.stdout.replace("\n", "")


# Toolkit user identity (extracted automatically)
TOOLKIT_USER_ORG = _run_shell_cmd("eai organization get --field name")
TOOLKIT_USER_ACCOUNT = _run_shell_cmd("eai account get --field name")
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
TOOLKIT_CPU = 2
TOOLKIT_GPU = 1
TOOLKIT_MEM = 32


def _load_envs():
    """Load environment variables that must be set in toolkit"""
    return [env.strip() for env in open(".envs", "r")]


def toolkit_job(script: Path, env_vars=()):
    """Launch a job on toolkit along with a specific script (assumed runnable with sh)"""
    job_name = (
        script.parent.name.lower()
    )  # TODO actually job_name needs to be unique across all jobs? maybe we need to rethink that.
    for char in (".", "="):  # TODO replace all non alpha numeric chars by '_'.
        job_name = job_name.replace(char, "_")

    # General job config
    cmd = f"eai job new -i {TOOLKIT_IMAGE} --non-preemptable".split(" ")

    # Set job name
    # cmd += ["--name", job_name]  # TODO: Job names cause issues and are not super useful

    # Computational requirements
    cmd += ["--cpu", str(TOOLKIT_CPU)]
    cmd += ["--gpu", str(TOOLKIT_GPU)]
    cmd += ["--mem", str(TOOLKIT_MEM)]

    # Mount data objects
    cmd += ["--data", f"{TOOLKIT_DATA}:/mnt/data"]
    cmd += ["--data", f"{TOOLKIT_CODE}@{TOOLKIT_CODE_VERSION}:/mnt/code"]

    # Set all environment variables
    for e in TOOLKIT_ENVS + env_vars:
        cmd += ["--env", e]

    # TODO: faire poetry install on boot
    cmd += [
        "--",
        f"sh -c '{TOOLKIT_BOOTSTRAP_CMD} && cd /mnt/code/ && sh {str(script)}'",
    ]  # TODO do we have to put absolute path? Could we use relative path of script?

    # Launch the job
    print(" ".join(cmd))
    _run_shell_cmd(" ".join(cmd))
    print("Launched.")


def toolkit_dispatcher(exp_dir, prompt=True, env_vars=()):
    """Scans the exp_dir for experiments to launch"""
    exp_dir = Path(exp_dir)

    print(f"Scanning {exp_dir}.")
    script_list = list(exp_dir.glob("**/run.sh"))
    if prompt:
        print("Will launch all of these jobs on toolkit:")
        for script in script_list:
            print(str(script))
        ans = input("Ready to proceed? y/n.")
        if ans != "y":
            return

    for script in script_list:
        toolkit_job(script, env_vars)

    print("Done.")


def push_code(dir):
    """Push the local code to the cluster"""
    print("Pushing code...")
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
