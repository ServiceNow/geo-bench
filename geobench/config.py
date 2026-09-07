"""Resolution of the geobench data directory, from $GEO_BENCH_DIR or a default."""

import os
from pathlib import Path

_GEO_BENCH_DIR_ENV = os.environ.get("GEO_BENCH_DIR", None)

if _GEO_BENCH_DIR_ENV is None:
    GEO_BENCH_DIR = Path("~").expanduser() / "dataset" / "geobench"
else:
    GEO_BENCH_DIR = Path(_GEO_BENCH_DIR_ENV)
