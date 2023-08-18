import os
from pathlib import Path

_GEO_BENCH_DIR = os.environ.get("GEO_BENCH_DIR", None)

if _GEO_BENCH_DIR is None:
    GEO_BENCH_DIR = Path("~").expanduser() / "dataset" / "geobench"
else:
    GEO_BENCH_DIR = Path(_GEO_BENCH_DIR)

GEO_BENCH_DIR.mkdir(exist_ok=True, parents=True)