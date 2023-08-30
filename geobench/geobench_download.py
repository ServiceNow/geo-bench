#!/usr/bin/env python

from pathlib import Path
import requests

from tqdm import tqdm
from geobench import config
import zipfile
import hashlib
import time

BENCHMARK_MAP = {
    "classification": [
        "m-bigearthnet",
        "m-brick-kiln",
        "m-eurosat",
        "m-forestnet",
        "m-pv4ger",
        "m-so2sat",
    ],
    "segmentation": [
        "m-chesapeake-lancover",
        "m-NeonTree",
        "m-nz-cattle",
        "m-pv4ger-seg",
        "m-cashew-plantation",
        "m-SA-crop-type",
    ],
}


IDENTIFIERS = {
    "m-so2sat": "8276566",
    "m-SA-crop-type": "8277133",
    "m-forestnet": "8277532",
    "m-NeonTree": "8277543",
    "m-bigearthnet": "8277476",
    "m-pv4ger-seg": "8277053",
    "m-cashew-plantation": "8277065",
    "m-chesapeake-lancover": "8276992",
    "m-nz-cattle": "8277048",
    "m-pv4ger": "8276974",
    "m-eurosat": "8276932",
    "m-brick-kiln": "8276814",
    "version": "v0.9.1",
}


def download_zenodo_file(file_url: str, output_path: Path, checksum: str = None, n_retry=4):
    """Download a file from Zenodo and check the checksum. If the file already exists and the checksum is correct, skip the download."""
    output_path = Path(output_path)
    for attempt in range(n_retry):
        try:
            if file_exists_and_valid(output_path, checksum):
                tqdm.write(f"File {output_path} already exists and checksum is good.")
                return

            download_file(file_url, output_path, checksum)
            tqdm.write(f"Downloaded file: {file_url}")
            return
        except Exception as error:
            tqdm.write(f"Attempt {attempt + 1} failed for file: {output_path}. Error: {error}")

            # sleep 5 seconds before retrying
            time.sleep(10)

            if attempt + 1 == n_retry:
                raise error


def file_exists_and_valid(output_path: Path, checksum: str):
    """Check if the file exists and the checksum is correct."""
    if output_path.exists() and checksum:
        return file_checksum(output_path) == checksum
    return False


def file_checksum(output_path: Path):
    """Compute the MD5 checksum of a file."""
    with output_path.open("rb") as file:
        file_hash = hashlib.md5()
        while chunk := file.read(8192):
            file_hash.update(chunk)
        return f"md5:{file_hash.hexdigest()}"


def _get_zenodo_record_by_url(url_or_identifier):
    identifier = url_or_identifier.split("/")[-1]
    record = requests.get(f"https://zenodo.org/api/records/{identifier}")
    record.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    return record.json()


def get_zenodo_record_by_url(url_or_identifier, retry=4):
    """Get the record data from a Zenodo URL."""
    for attempt in range(retry):
        try:
            return _get_zenodo_record_by_url(url_or_identifier)
        except Exception as error:
            tqdm.write(f"Attempt {attempt + 1} failed for url: {url_or_identifier}. Error: {error}")

            # sleep 5 seconds before retrying
            time.sleep(5)

            if attempt + 1 == retry:
                raise error


def download_file(file_url, output_path, checksum=None):
    """Download a file from a URL and check the checksum."""
    response = requests.get(file_url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    descr = "/".join(output_path.parts[-3:])
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc=descr, leave=False)

    with output_path.open("wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            progress_bar.update(len(chunk))

    progress_bar.close()

    if checksum and file_checksum(output_path) != checksum:
        raise ValueError(f"Checksum mismatch for file: {file_url}")


def _extract_flat_zip(zip_file):
    """Extract all zip file to the same root directory.

    This is temporary due to a bug in the upload
    """
    with zipfile.ZipFile(zip_file) as zip:
        for zip_info in zip.infolist():
            if zip_info.is_dir():
                continue
            zip_info.filename = Path(zip_info.filename).name
            zip.extract(zip_info, Path(zip_file).parent)


def download_dataset(files: list, dataset_dir: str):
    """Download a dataset from Zenodo.

    Parameters:
        files (list): A list of files to download. Each file is a dict with keys "key", "links", and "checksum".
        dataset_dir (str): The directory to download the dataset to.

    """
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    done_path = dataset_dir / "done.txt"
    if done_path.exists():
        tqdm.write(f"Dataset {dataset_dir} already downloaded.")
        return

    for file in files:
        output_path = dataset_dir / file["key"]
        url = file["links"]["self"]
        download_zenodo_file(url, output_path=output_path, checksum=file["checksum"])
        time.sleep(1)  # to reduce changes of hitting rate limit
        if output_path.suffix == ".zip":
            _extract_flat_zip(output_path)
            output_path.unlink()

    done_path.touch()


def download_benchmark(
    version="v0.9.1",
    benchmark_names=("classification", "segmentation"),
    geobench_dir=None,
    parallel=False,
):
    """Download geobench from Zenodo.

    Parameters:
        version (str): The version of the benchmark to download. Version 1.0.0 coming soon.
        benchmark_names (tuple): The benchmarks to download. Can be "classification" or "segmentation".
        geobench_dir (str): The directory to download the benchmark to. If None, use GEO_BENCH_DIR.
    """
    if geobench_dir is None:
        geobench_dir = config.GEO_BENCH_DIR
    geobench_dir = Path(geobench_dir)

    assert IDENTIFIERS["version"] == version
    all_files = []

    for benchmark_name in benchmark_names:
        print(f"Collecting records for geobench: {benchmark_name} {version} to {geobench_dir}")

        for dataset_name in BENCHMARK_MAP[benchmark_name]:
            record = get_zenodo_record_by_url(IDENTIFIERS[dataset_name])
            dataset_dir = geobench_dir / f"{benchmark_name}_{version}" / dataset_name
            all_files.append((record["files"], dataset_dir))

    if parallel:
        from tqdm.contrib.concurrent import thread_map

        thread_map(lambda tuple: download_dataset(*tuple), all_files)
    else:
        for files, dataset_dir in all_files:
            download_dataset(files, dataset_dir)


def get_zenodo_records(version="v0.9.1", community="geo-bench"):
    """Get all records from a Zenodo community. This is used to get the identifiers of all datasets."""

    # Browse all records in the geo-bench community
    from sickle import Sickle

    sickle = Sickle("https://zenodo.org/oai2d")

    records = {}
    for record in sickle.ListRecords(metadataPrefix="oai_dc", set=f"user-{community}"):
        # Extract the record identifier
        identifier = [
            id.replace("https://zenodo.org/record/", "")
            for id in record.metadata["identifier"]
            if id.startswith("https://zenodo.org/record/")
        ][0]
        record = get_zenodo_record_by_url(identifier)

        # Keep only records that have the correct version
        if version is not None:
            if record["metadata"]["version"] != version:
                continue

        # Download data
        dataset_name = record["metadata"]["title"].split(":")[1].strip()

        records[dataset_name] = record

    print({dataset_name: record["conceptrecid"] for dataset_name, record in records.items()})

    return records


if __name__ == "__main__":
    download_benchmark()

    # # for developers
    # download_benchmark(geobench_dir=Path("~/geobench_test").expanduser())
    # get_zenodo_records()
