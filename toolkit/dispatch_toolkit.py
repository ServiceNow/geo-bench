import argparse
from pathlib import Path
import subprocess

from subprocess import PIPE


def _run_shell_cmd(cmd: str, hide_stderr=False):
    """Run a shell command and return the output"""
    result = subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE if hide_stderr else None, universal_newlines=True)
    return result.stdout.replace("\n", "")


# Toolkit user identity (extracted automatically)
TOOLKIT_USER_ORG = _run_shell_cmd("eai organization get --field name")
TOOLKIT_USER_ACCOUNT = _run_shell_cmd("eai account get --field name")
TOOLKIT_USER = f"{TOOLKIT_USER_ORG}.{TOOLKIT_USER_ACCOUNT}"

# Infrastructure setup variables
# Note: Give access to new users by adding them to the eai.rg_climate_benchmark.admin role.
TOOLKIT_IMAGE = "registry.console.elementai.com/snow.rg_climate_benchmark/base:main"
TOOLKIT_DATA = "snow.rg_climate_benchmark.data"
TOOLKIT_CODE = "snow.rg_climate_benchmark.code"
TOOLKIT_BOOTSTRAP_CMD = "cd /mnt/code/ && poetry build && pip install dist/climate_change_benchmark-*.whl && export PATH=$PATH:/tmp/.local/bin"


def _load_envs():
    """Load environment variables that must be set in toolkit"""
    return [env.strip() for env in open(".envs", "r")]


def toolkit_job(script: Path):
    job_name = (
        script.parent.name.lower()
    )  # TODO actually job_name needs to be unique across all jobs? maybe we need to rethink that.
    for char in (".", "="):  # TODO replace all non alpha numeric chars by '_'.
        job_name = job_name.replace(char, "_")

    # General job config
    cmd = f"eai job new -i {TOOLKIT_IMAGE} --non-preemptable".split(" ")

    # Set job name
    # cmd += ["--name", job_name]  # TODO: causes issues

    # Mount data objects
    cmd += ["--data", f"{TOOLKIT_DATA}:/mnt/data"]
    cmd += ["--data", f"{TOOLKIT_CODE}@{TOOLKIT_USER}:/mnt/code"]

    # Set all environment variables
    for e in _load_envs():
        cmd += ["--env", e]

    # TODO: faire poetry install on boot
    cmd += [
        "--",
        # f"sh -c '{TOOLKIT_BOOTSTRAP_CMD} && {str(script)}'"
        f"sh -c '{TOOLKIT_BOOTSTRAP_CMD}'",
    ]  # TODO do we have to put absolute path? Could we use relative path of script?

    # Launch the job
    print(" ".join(cmd))
    _run_shell_cmd(" ".join(cmd))
    print("Launched.")


def toolkit_dispatcher(exp_dir, prompt=True):
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
        toolkit_job(script)

    print("Done.")


def start():
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="sequential_dispatcher.py",
        description="Sequentially dispatch all run.sh in the experiment directory.",
    )

    parser.add_argument(
        "--experiment-dir",
        help="The based directory in which experiment-related files should be created.",
        required=True,
    )

    args = parser.parse_args()

    # TODO push code in user specific directory

    toolkit_dispatcher(args.experiment_dir)


def push_code():
    """Push the local code to the cluster
    :param toolkit: contains value related to toolkit (e.g. user_account)
    """

    print("Pushing code...")
    _run_shell_cmd(f"eai data branch add {TOOLKIT_CODE}@empty {TOOLKIT_USER}")
    # TODO: path is .. since we are in the toolkit dir. Could make cleaner later
    cmd = f" rsync -a .. /tmp/rg_climate_benchmark --delete --exclude-from='.eaiignore' && \
           eai data push {TOOLKIT_CODE}@{TOOLKIT_USER} /tmp/rg_climate_benchmark:/ && \
           rm -rf /tmp/rg_climate_benchmark"
    _run_shell_cmd(cmd)


if __name__ == "__main__":
    # start()
    push_code()
    toolkit_job(Path("/bla"))
