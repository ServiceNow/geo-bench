# TODO the tqdm process bars will not work properly. Perhaps, we should revert to simple logging instead of tqdm + prints.

import multiprocessing
import shutil
from ccb import io
from pathlib import Path
from importlib import import_module

CONVERTERS = ["brick_kiln", "neon_tree", "cv4a_kenya_crop_type", "benin_smallholder_cashews", "eurosat"]

MAX_COUNT = 1000


def convert(module_name):
    converter = import_module("ccb.dataset_converters." + module_name)
    assert Path(converter.DATASET_DIR).parent == Path(
        io.datasets_dir), f"{Path(converter.DATASET_DIR).parent} vs {io.datasets_dir}"
    assert Path(converter.DATASET_DIR).name == converter.DATASET_NAME

    shutil.rmtree(converter.DATASET_DIR)
    converter.convert(max_count=MAX_COUNT)


if __name__ == "__main__":

    response = input(f"This will first delete all datasets in {io.datasets_dir}. To proceed, press 'y'.")
    if response.lower() == 'y':
        jobs = []
        for converter in CONVERTERS:
            job = multiprocessing.Process(target=convert, args=(converter,))
            jobs.append(job)
            job.start()

        for job in jobs:
            job.join()
    else:
        print("No dataset deleted.")
