import contextlib
import io

import pytest

from geobench.config import GEO_BENCH_DIR

BENCHMARKS = ("classification_v1.0", "segmentation_v1.0")


@pytest.mark.skipif(
    not all((GEO_BENCH_DIR / name).is_dir() for name in BENCHMARKS),
    reason=f"requires the downloaded benchmarks {BENCHMARKS} in {GEO_BENCH_DIR}",
)
def test_load_dataset():
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        # just importing is enough to run it
        pass

    output = captured_output.getvalue()

    for word in ["Task", "Sample", "band", "eurosat", "pv4ger"]:
        assert word in output, f"word {word} not found in output"


if __name__ == "__main__":
    test_load_dataset()
