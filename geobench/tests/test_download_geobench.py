from pathlib import Path
from geobench import download_geobench
import tempfile
from contextlib import redirect_stdout
from io import StringIO
import pytest


def test_download_zenodo_file():
    url = "https://zenodo.org/record/8263732/files/data.zip?download=1"
    checksum = "md5:2849dc86c9bc33cf0c6777b0ebfa1d8c"
    wrong_checksum = "wrong_checksum_value"

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as tmp_file:
        captured_output = StringIO()

        # assert download file works
        with redirect_stdout(captured_output):
            download_geobench.download_zenodo_file(url, tmp_file.name, checksum=checksum)

        # assert skip download on file exists
        with redirect_stdout(captured_output):
            download_geobench.download_zenodo_file(url, tmp_file.name, checksum=checksum)
        output = captured_output.getvalue()

    assert output.strip().endswith("already exists and checksum is good.")

    # assert raise on wrong checksum
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as tmp_file:
        with pytest.raises(Exception):
            with redirect_stdout(captured_output):
                download_geobench.download_zenodo_file(url, tmp_file.name, checksum=wrong_checksum)


def test_download_dataset():
    record = download_geobench.get_zenodo_record_by_url("https://zenodo.org/record/8263732")

    with tempfile.TemporaryDirectory() as tmp_dir:
        download_geobench.download_dataset(record["files"], tmp_dir)

        expected_files = [
            "id_99972.hdf5",
            "id_99978.hdf5",
            "label_map.json",
            "LICENSE",
            "README",
            "task_specs.pkl",
        ]

        tmp_path = Path(tmp_dir)
        actual_files = [f.name for f in tmp_path.iterdir()]

        for expected_file in expected_files:
            assert expected_file in actual_files, f"{expected_file} is missing in the directory"


if __name__ == "__main__":
    # test_download_zenodo_file()

    test_download_dataset()
