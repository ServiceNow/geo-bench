from pathlib import Path
from tqdm import tqdm
from contexttimer import Timer
import zipfile
import tarfile
import os


def decompress_zip_with_progress(zip_file_path, extract_to_folder):
    # Open the zip file
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        with Timer(prefix="get n-files", output=True) as t:
            # Get the list of file names in the zip
            file_names = zip_ref.namelist()
            # Calculate the total number of files
            total_files = len(file_names)

        # Initialize the progress bar with the total number of files
        with tqdm(total=total_files, unit="file", desc="Extracting") as pbar:
            for file in file_names:
                # Extract each file
                zip_ref.extract(file, extract_to_folder)
                # Update the progress bar
                pbar.update(1)


# Function to decompress tar.gz file
def decompress_tar(file_path):
    if file_path.suffix == ".gz":
        print(f"Decompressing {file_path}...")
        with tarfile.open(file_path, "r:gz") as tar:
            with Timer(prefix="get members", output=True) as t:
                members = tar.getmembers()
            for member in tqdm(members, desc=f"Decompressing {file_path.name}", leave=False):
                tar.extract(member, path=file_path.parent)


# Usage

zip_path = Path("/mnt/home/dataset/geobench-1.0/classification_v1.0/m-eurosat.5.zip")
tar_path = Path("/mnt/home/dataset/geobench-1.0/classification_v1.0/m-eurosat-lowest.tar.gz")
dst_dir = Path().home() / "dataset" / "test"
dst_dir.mkdir(parents=True, exist_ok=True)


with Timer(prefix="total untar time", output=True) as t:
    decompress_tar(tar_path)


with Timer(prefix="total unzip time", output=True) as t:
    decompress_zip_with_progress(zip_path, dst_dir)
