"""
Zenodo Uploader: Uploads the datasets to the Geo-Bench community on Zenodo.

By: Alexandre Drouin (ServiceNow Research)
Date: June 2, 2023

"""
import json
import os
import requests

from path import Path


ACCESS_TOKEN = os.environ["ZENODO_TOKEN"]

VERSION = "v0.9.1"

AUTHORS = [
    {"name": "Lacoste, Alexandre", "affiliation": "ServiceNow Research"},
    {"name": "Lehmann, Niels", "affiliation": "University of Amsterdam"},
    {"name": "Rodriguez, Pau", "affiliation": "ServiceNow Research"},
    {"name": "David Sherwin, Evan", "affiliation": "Stanford University"},
    {"name": "Kerner, Hannah", "affiliation": "University of Maryland"},
    {"name": "Lütjens, Björn", "affiliation": "Massachusetts Institute of Technology"},
    {"name": "Irvin, Jeremy", "affiliation": "Stanford University"},
    {"name": "Dao, David", "affiliation": "ETH Zurich"},
    {"name": "Alemohammad, Hamed", "affiliation": "Clark University"},
    {"name": "Drouin, Alexandre", "affiliation": "ServiceNow Research, Mila-Quebec"},
    {"name": "Gunturkun, Mehmet", "affiliation": "ServiceNow Research"},
    {"name": "Huang, Gabriel", "affiliation": "ServiceNow Research, University of Montreal"},
    {"name": "Vazquez, David", "affiliation": "ServiceNow Research"},
    {"name": "Newman, Dava", "affiliation": "Massachusetts Institute of Technology"},
    {"name": "Bengio, Yoshua", "affiliation": "Mila-Quebec, University of Montreal"},
    {"name": "Ermon, Stefano", "affiliation": "Stanford University"},
    {"name": "Xiang Zhu, Xiao", "affiliation": "Technical University of Munich"},
]

DATASETS = [
    # Classification datasets
    ("m-so2sat", f"classification_{VERSION}/so2sat", "cc-by-4.0"),
    ("m-bigearthnet", f"classification_{VERSION}/bigearthnet", "cdla-permissive-1.0"),
    ("m-brick-kiln", f"classification_{VERSION}/brick_kiln_v1.0", "cc-by-sa-4.0"),
    ("m-forestnet", f"classification_{VERSION}/forestnet_v1.0", "cc-by-4.0"),
    ("m-eurosat", f"classification_{VERSION}/eurosat", "cc-by-4.0"),
    ("m-pv4ger", f"classification_{VERSION}/pv4ger_classification", "mit"),
    # Segmentation datasets
    (
        "m-chesapeake-lancover",
        f"segmentation_{VERSION}/cvpr_chesapeake_landcover",
        "cdla-permissive-1.0",
    ),
    ("m-NeonTree", f"segmentation_{VERSION}/NeonTree_segmentation", "cc-by-4.0"),
    ("m-nz-cattle", f"segmentation_{VERSION}/nz_cattle_segmentation", "cc-by-4.0"),
    ("m-pv4ger-seg", f"segmentation_{VERSION}/pv4ger_segmentation", "mit"),
    ("m-cashew-plantation", f"segmentation_{VERSION}/smallholder_cashew", "cc-by-4.0"),
    ("m-SA-crop-type", f"segmentation_{VERSION}/southAfricaCropType", "cc-by-4.0")
    # Test dataset
    ("test-dataset", f"classification_test_{VERSION}/test_dataset", "mit"),
]

fail_log = open("failed.log", "w")
for dataset_name, dataset_directory, dataset_license in DATASETS:
    try:
        print("Dataset:", dataset_name)

        dataset_directory = Path(dataset_directory)

        # Create data bucket
        print("Creating data bucket...")
        headers = {"Content-Type": "application/json"}
        params = {"access_token": ACCESS_TOKEN}
        r = requests.post(
            "https://zenodo.org/api/deposit/depositions",
            params=params,
            json={},
            headers=headers,
        )
        assert r.status_code == 201, "Error creating bucket: %s" % r.status_code
        bucket_url = r.json()["links"]["bucket"]
        bucket_id = r.json()["id"]

        print("Setting metadata...")
        data = {
            "metadata": {
                "title": f"GEO-Bench: {dataset_name}",
                "upload_type": "dataset",
                "description": open(f"{dataset_directory}/README", "r").read(),
                "creators": AUTHORS,
                "communities": [{"identifier": "geo-bench"}]
                if dataset_name != "test-dataset"
                else [],
                "prereserve_doi": True,
                "version": VERSION,
                "license": dataset_license,
            }
        }
        r = requests.put(
            "https://zenodo.org/api/deposit/depositions/%s" % bucket_id,
            params={"access_token": ACCESS_TOKEN},
            data=json.dumps(data),
            headers=headers,
        )
        assert r.status_code == 200, "Error adding metadata: %s" % r.status_code

        # Upload files
        print("Zipping files...")
        pwd = Path(os.path.curdir).abspath()
        os.system(f"cd {dataset_directory}; rm data.zip; zip data.zip ./*.hdf5; cd {pwd}")

        print("Fetching files to upload...")
        files_to_upload = [
            dataset_directory / "data.zip",
            dataset_directory / "LICENSE",
            dataset_directory / "README",
        ]
        # Upload all json and pickle files too
        files_to_upload += list(Path(dataset_directory).glob("*.json"))
        files_to_upload += list(Path(dataset_directory).glob("*.pkl"))

        print("Uploading files...")
        for i, fpath in enumerate(files_to_upload):
            filename = fpath.name

            print(f"... file {filename} ({i+1}/{len(files_to_upload)})")

            if not fpath.exists():
                print("File %s does not exist, skipping" % fpath)
                continue

            with open(fpath, "rb") as fp:
                r = requests.put(
                    "%s/%s" % (bucket_url, filename),
                    data=fp,
                    params={"access_token": ACCESS_TOKEN},
                )
                assert r.status_code == 200, "Error uploading file: %s" % r.status_code

    except Exception as e:
        fail_log.write(f"{dataset_name}: {e}\n")
        print(f"Failed to upload {dataset_name}: {e}")
