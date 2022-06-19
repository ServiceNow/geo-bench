"""Sequential dispatcher."""
import argparse
import subprocess
from pathlib import Path

from ccb.experiment.experiment import Job


def sequential_dispatcher(exp_dir: str, prompt: bool = True) -> None:
    """Dispatch a series of jobs in sequential manner.

    Args:
        exp_dir: path to experiment dir containing job directories
        prompt: whether or not to prompt the user for execution of scripts
    """
    exp_dir = Path(exp_dir)

    print(f"Scanning {exp_dir}.")
    script_list = list(exp_dir.glob("**/run.sh"))
    if prompt:
        print("Will sequentially execute all of these scripts:")
        for script in script_list:
            print(str(script))
        ans = input("ready to proceed? y/n.")
        if ans != "y":
            return

    for script in script_list:
        print(f"Running {script}.")
        job = Job(script.parent)

        subprocess.run([script])
        print(job.get_stderr())

    print("Done.")


def start():
    """Start sequential dispatcher."""
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="sequential_dispatcher.py", description="Sequentially dispatch all run.sh in the experiment directory."
    )

    parser.add_argument(
        "--experiment-dir",
        help="The based directory in which experiment-related files should be created.",
        required=True,
    )

    args = parser.parse_args()
    sequential_dispatcher(args.experiment_dir)


if __name__ == "__main__":
    start()
