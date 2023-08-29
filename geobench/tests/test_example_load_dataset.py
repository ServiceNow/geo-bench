import io
import contextlib


def test_load_dataset():
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        # just importing is enough to run it
        from geobench import example_load_datasets

    output = captured_output.getvalue()

    for word in ["Task", "Sample", "band", "eurosat", "pv4ger"]:
        assert word in output, f"word {word} not found in output"


if __name__ == "__main__":
    test_load_dataset()
