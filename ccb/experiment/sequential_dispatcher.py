import argparse
from pathlib import Path
import subprocess


def sequential_dispatcher(exp_dir, prompt=True):
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
        subprocess.check_call([script])

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
    sequential_dispatcher(args.experiment_dir)


if __name__ == "__main__":
    start()
