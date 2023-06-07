"""
Download the GEO-Bench benchmark from Zenodo

Before running the script, make sure to install the following packages:

```
pip install requests
pip install sickle
pip install tqdm
```

"""
import argparse
import os

try:
    import requests
except:
    print("Please install the requests package and try again: 'pip install requests'.")

try:
    from sickle import Sickle
except:
    print("Please install the sickle package and try again: 'pip install sickle'.")

try:
    from tqdm import tqdm
except:
    print("Please install the tqdm package and try again: 'pip install tqdm'.")


COMMUNITY = "geo-bench"


def download_zenodo_record(record_data, output_dir):
    """
    Download the files from a Zenodo record

    Parameters
    ----------
    record_data : dict
        Dictionary containing the record data (retrieved from the Zenodo API)
    output_dir : str
        Output directory to save the files

    Returns
    -------
    None

    """
    # Extract file information
    files = record_data["files"]

    # Download each file
    for file in files:
        file_url = file["links"]["self"]
        file_name = file["key"]
        output_path = f"{output_dir}/{file_name}"

        # Download the file
        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))

            progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc=file_name, leave=True)
            with open(output_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    progress_bar.update(len(chunk))
            print(f"Downloaded file: {file_name}")
        else:
            print(f"Failed to download file: {file_name}. Status code: {response.status_code}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download the GEO-Bench benchmark from Zenodo")
    parser.add_argument("-v", "--version", type=str, help="Version of the data to download", required=True)
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory to save the data", default=".")
    args = parser.parse_args()

    # Initialize the Sickle client
    sickle = Sickle("https://zenodo.org/oai2d")

    # Browse all records in the geo-bench community
    for record in sickle.ListRecords(metadataPrefix="oai_dc", set=f"user-{COMMUNITY}"):

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
        if record["metadata"]["version"] != args.version:
            continue

        # Download data
        dataset_name = record["metadata"]["title"].split(":")[1].strip()
        print("> Downloading dataset:", dataset_name)
        out_path = os.path.join(args.output_dir, dataset_name)
        os.makedirs(out_path, exist_ok=True)
        download_zenodo_record(record, out_path)

print("Completed.")
