from ccb import io
from pathlib import Path
import tempfile
import pytest


def converter_tester(converter):
    assert "convert" in dir(converter)
    assert "DATASET_NAME" in dir(converter)
    assert "DATASET_DIR" in dir(converter)
    with tempfile.TemporaryDirectory() as datasets_dir:
        dataset_dir = Path(datasets_dir, converter.DATASET_NAME)
        converter.convert(max_count=5, dataset_dir=Path(dataset_dir))
        dataset = io.Dataset(dataset_dir)
        assert len(dataset) == 5
        io.check_dataset_integrity(dataset)


SRC_DIR_EXISTS = not Path(io.src_datasets_dir).exists()


@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_brick_kiln():
    from ccb.dataset_converters import brick_kiln

    converter_tester(brick_kiln)


@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_cv4a_kenya_cropy_type():
    from ccb.dataset_converters import cv4a_kenya_crop_type

    converter_tester(cv4a_kenya_crop_type)


@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_eurosat():
    from ccb.dataset_converters import eurosat

    converter_tester(eurosat)


@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_neon_tree():
    from ccb.dataset_converters import neon_tree

    converter_tester(neon_tree)


@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_smallholder_cashews():
    from ccb.dataset_converters import benin_smallholder_cashews

    converter_tester(benin_smallholder_cashews)


@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_so2sat():
    from ccb.dataset_converters import so2sat

    converter_tester(so2sat)


@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_nz_cattle_detection():
    from ccb.dataset_converters import nz_cattle_detection

    converter_tester(nz_cattle_detection)
