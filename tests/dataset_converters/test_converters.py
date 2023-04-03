import tempfile
from pathlib import Path

import pytest

from geobench import io


def converter_tester(converter):
    assert "convert" in dir(converter)
    assert "DATASET_NAME" in dir(converter)
    assert "DATASET_DIR" in dir(converter)
    with tempfile.TemporaryDirectory() as datasets_dir:
        dataset_dir = Path(datasets_dir, converter.DATASET_NAME)
        converter.convert(max_count=5, dataset_dir=Path(dataset_dir))
        dataset = io.GeobenchDataset(dataset_dir, band_names=["red", "green", "blue"], partition_name="default")
        samples = list(dataset.iter_dataset(5))
        assert len(dataset) == 5, f"returned dataset of length {len(dataset)}"
        io.check_dataset_integrity(dataset, samples=samples)


SRC_DIR_EXISTS = not Path(io.src_datasets_dir).exists()


@pytest.mark.converter
@pytest.mark.slow
@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_brick_kiln():
    from geobench.benchmark.dataset_converters import brick_kiln

    converter_tester(brick_kiln)


# @pytest.mark.converter
# @pytest.mark.slow
# @pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
# def test_cv4a_kenya_cropy_type():
#     from geobench.benchmark.dataset_converters import cv4a_kenya_crop_type

#     converter_tester(cv4a_kenya_crop_type)


@pytest.mark.converter
@pytest.mark.slow
@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_eurosat():
    from geobench.benchmark.dataset_converters import eurosat

    converter_tester(eurosat)


@pytest.mark.converter
@pytest.mark.slow
@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_neon_tree():
    from geobench.benchmark.dataset_converters import neon_tree

    converter_tester(neon_tree)


@pytest.mark.converter
@pytest.mark.slow
@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_smallholder_cashews():
    from geobench.benchmark.dataset_converters import benin_smallholder_cashews

    converter_tester(benin_smallholder_cashews)


@pytest.mark.converter
@pytest.mark.slow
@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_so2sat():
    from geobench.benchmark.dataset_converters import so2sat

    converter_tester(so2sat)


@pytest.mark.converter
@pytest.mark.slow
@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_geolifeclef():
    from geobench.benchmark.dataset_converters import geolifeclef

    converter_tester(geolifeclef)


@pytest.mark.converter
@pytest.mark.slow
@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_nz_cattle_detection():
    from geobench.benchmark.dataset_converters import nz_cattle_detection

    converter_tester(nz_cattle_detection)


@pytest.mark.converter
@pytest.mark.slow
@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_xview2():
    from geobench.benchmark.dataset_converters import xview2

    converter_tester(xview2)


@pytest.mark.converter
@pytest.mark.slow
@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_pv4ger():
    from geobench.benchmark.dataset_converters import pv4ger

    converter_tester(pv4ger)


@pytest.mark.converter
@pytest.mark.slow
@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_chesapeake():
    from geobench.benchmark.dataset_converters import chesapeake_land_cover

    converter_tester(chesapeake_land_cover)


@pytest.mark.converter
@pytest.mark.slow
@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_forestnet():
    from geobench.benchmark.dataset_converters import forestnet

    converter_tester(forestnet)


@pytest.mark.converter
@pytest.mark.slow
@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_bigearthnet():
    from geobench.benchmark.dataset_converters import bigearthnet

    converter_tester(bigearthnet)


@pytest.mark.converter
@pytest.mark.slow
@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_south_africa_crop_type():
    from geobench.benchmark.dataset_converters import crop_type_south_africa

    converter_tester(crop_type_south_africa)

@pytest.mark.converter
@pytest.mark.slow
@pytest.mark.skipif(SRC_DIR_EXISTS, reason="Requires presence of the source datasets.")
def test_seasonet():
    from geobench.benchmark.dataset_converters import seasonet

    converter_tester(seasonet)


if __name__ == "__main__":
    # test_brick_kiln()
    # test_cv4a_kenya_cropy_type()
    test_eurosat()
    # test_neon_tree()
    # # test_smallholder_cashews()
    # test_so2sat()
    # test_nz_cattle_detection()
    # test_xview2()
    # test_bigearthnet()
    # test_south_africa_crop_type()
    # test_chesapeake()
    # test_forestnet()
    # test_pv4ger()
    # test_seasonet()
