
import datetime
from ccb.io.dataset import Dataset, check_dataset_integrity
from pathlib import Path
import tempfile


def converter_tester(converter):
    assert 'convert' in dir(converter)
    assert 'DATASET_NAME' in dir(converter)
    assert 'DATASET_DIR' in dir(converter)
    with tempfile.TemporaryDirectory() as datasets_dir:
        dataset_dir = Path(datasets_dir, converter.DATASET_NAME)
        converter.convert(max_count=5, dataset_dir=Path(dataset_dir))
        dataset = Dataset(dataset_dir)
        assert len(dataset) == 5
        check_dataset_integrity(dataset)


def test_brick_kiln():
    from ccb.dataset_converters import brick_kiln
    converter_tester(brick_kiln)


def test_cv4a_kenya_cropy_type():
    from ccb.dataset_converters import cv4a_kenya_crop_type
    converter_tester(cv4a_kenya_crop_type)


def test_eurosat():
    from ccb.dataset_converters import eurosat
    converter_tester(eurosat)


def test_neon_tree():
    from ccb.dataset_converters import neon_tree
    converter_tester(neon_tree)


def test_smallholder_cashews():
    from ccb.dataset_converters import benin_smallholder_cashews
    converter_tester(benin_smallholder_cashews)
