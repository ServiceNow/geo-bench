from pathlib import Path
import requests

from sickle import Sickle
from tqdm import tqdm
from geobench import config
from tqdm.contrib.concurrent import thread_map
import zipfile
import hashlib


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
        # "m-chesapeake-lancover",
        # "m-NeonTree",
        "m-nz-cattle",
        # "m-pv4ger-seg",
        # "m-cashew-plantation",
        # "m-SA-crop-type",
    ],
}


def md5_checksum(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as file:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: file.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()


def _extract_flat_zip(zip_file):
    with zipfile.ZipFile(zip_file) as zip:
        for zip_info in zip.infolist():
            if zip_info.is_dir():
                continue
            zip_info.filename = Path(zip_info.filename).name
            zip.extract(zip_info, Path(zip_file).parent)


def download_zenodo_file(file_url, output_path):
    response = requests.get(file_url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))

        # take the last 3 elements of the output path
        descr = "/".join(output_path.parts[-3:])

        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc=descr, leave=False)
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress_bar.update(len(chunk))
        progress_bar.close()

        print(f"Downloaded file: {file_url}")
    else:
        print(f"Failed to download file: {file_url}. Status code: {response.status_code}")


# def download_zenodo_record(record_data, output_dir):
#     """
#     Download the files from a Zenodo record

#     Parameters
#     ----------
#     record_data : dict
#         Dictionary containing the record data (retrieved from the Zenodo API)
#     output_dir : str
#         Output directory to save the files

#     Returns
#     -------
#     None

#     """
#     # Extract file information
#     files = record_data["files"]

#     # Download each file
#     for file in files:
#         file_url = file["links"]["self"]
#         file_name = file["key"]
#         output_path = f"{output_dir}/{file_name}"

#         # Download the file
#         response = requests.get(file_url, stream=True)
#         if response.status_code == 200:
#             total_size = int(response.headers.get("content-length", 0))

#             progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc=file_name, leave=True)
#             with open(output_path, "wb") as file:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     file.write(chunk)
#                     progress_bar.update(len(chunk))
#             print(f"Downloaded file: {file_name}")
#         else:
#             print(f"Failed to download file: {file_name}. Status code: {response.status_code}")


def download_dataset(files, dataset_dir):
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        output_path = dataset_dir / file["key"]
        url = file["links"]["self"]
        download_zenodo_file(url, output_path=output_path)

        md5 = md5_checksum(file_path=output_path)
        if md5 != file["checksum"]:
            raise Exception(f"MD5 checksum failed for file: {output_path}")

        if output_path.suffix == ".zip":
            _extract_flat_zip(output_path)
            output_path.unlink()

    (dataset_dir / "done.txt").touch()


def download_benchmark(version="v0.9.0", benchmark_namse=("classification", "segmentation"), geobench_dir=None):

    if geobench_dir is None:
        geobench_dir = config.GEO_BENCH_DIR
    geobench_dir = Path(geobench_dir)

    records = get_zenodo_records(version=version)
    all_files = []

    for benchmark_name in benchmark_namse:
        print(f"Downloading geobench: {benchmark_name} {version} to {geobench_dir}")

        for dataset_name in BENCHMARK_MAP[benchmark_name]:
            record = records[dataset_name]
            dataset_dir = geobench_dir / f"{benchmark_name}_{version}" / dataset_name
            all_files.append((record["files"], dataset_dir))

    thread_map(lambda tuple: download_dataset(*tuple), all_files)


def get_zenodo_records(version="v0.9.0", community="geo-bench"):
    # Browse all records in the geo-bench community

    sickle = Sickle("https://zenodo.org/oai2d")

    records = {}
    for record in sickle.ListRecords(metadataPrefix="oai_dc", set=f"user-{community}"):

        # Extract the record identifier
        identifier = [
            id.replace("https://zenodo.org/record/", "")
            for id in record.metadata["identifier"]
            if id.startswith("https://zenodo.org/record/")
        ][0]

        # Fetch the record data
        r = requests.get(f"https://zenodo.org/api/records/{identifier}")
        if r.status_code != 200:
            print("Error: Failed to fetch record:", identifier)
            exit()
        record = r.json()

        # Keep only records that have the correct version
        if version is not None:
            if record["metadata"]["version"] != version:
                continue

        # Download data
        dataset_name = record["metadata"]["title"].split(":")[1].strip()

        records[dataset_name] = record

    return records


if __name__ == "__main__":

    # record = get_zenodo_records()
    download_benchmark(geobench_dir=Path("~/geobench_test").expanduser())
