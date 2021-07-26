import multiprocessing
import time
import subprocess
import os
import shutil


# eai job new --image registry.console.elementai.com/snow.rg_climate_benchmark/base:main --data 1bc862ee-73d8-401a-bee6-942ba0580cae:/mnt -- ls | awk '{ print $1 }' | sed -n 2p
# returns job id
def next_available_dir():
    i = 0
    while os.path.exists(i):
        i += 1

    return i


def wait_until_exists(p):
    while True:
        if os.path.exists(p):
            return
        time.sleep(2)


def hp_search(fn):

    i = 0

    # out_dir = str(next_available_dir())
    out_dir = "logs"

    cmd = " ".join(("sh", fn, out_dir))

    subprocess.run(cmd, capture_output=True, shell=True)

    n_cmd = len(open(fn).readlines()) - 6
    print(n_cmd)

    for i in range(n_cmd):
        wait_until_exists(os.path.join(out_dir, str(i), "max_val_acc"))

    # TODO: produce table
    for i in range(n_cmd):
        with open(os.path.join(out_dir, str(i), "max_val_acc")) as f:
            print(i, f.readlines()[0])


if __name__ == "__main__":

    shutil.rmtree("logs")
    os.makedirs("logs", exist_ok=True)
    hp_search("hp_search.sh")
