"""Restricted unpickling of geobench metadata.

Band metadata and task specifications are stored as pickles inside GeoTIFF tags,
HDF5 attributes, npz entries and ``task_specs.pkl``. Unpickling executes arbitrary
code, so loading a sample from an untrusted source would hand that source control
of the interpreter (CWE-502).

:class:`SafeUnpickler` reconstructs only the small, fixed set of types geobench
actually writes and refuses everything else. It reads every dataset published for
GEO-Bench v1.0, so existing data does not need to be regenerated.
"""

import io
import pickle
from typing import IO, Any, Callable, Dict, Tuple

import numpy as np


class UnsafePickleError(pickle.UnpicklingError):
    """Raised when a pickle refers to a type outside the geobench allowlist."""


def _safe_dtype(*args, **kwargs) -> np.dtype:
    """Build a dtype, rejecting object dtypes.

    Object dtypes are the entry point for the ``numpy.dtype('O')`` gadget: numpy
    itself calls ``pickle.loads`` to restore object-dtype values, which would
    bypass this unpickler entirely.
    """
    dtype = np.dtype(*args, **kwargs)
    if dtype.hasobject:
        raise UnsafePickleError("object dtypes are not allowed")
    return dtype


def _safe_scalar(dtype, obj) -> Any:
    """Restore a numpy scalar, rejecting object dtypes."""
    if np.dtype(dtype).hasobject:
        raise UnsafePickleError("object-dtype scalars are not allowed")
    return _np_multiarray().scalar(dtype, obj)


def _safe_reconstruct(subtype, shape, dtype) -> np.ndarray:
    """Restore an ndarray, rejecting subclasses that could carry custom code."""
    if subtype is not np.ndarray:
        raise UnsafePickleError(f"ndarray subclass {subtype!r} is not allowed")
    return _np_multiarray()._reconstruct(subtype, shape, dtype)


def _np_multiarray():
    """Return numpy's multiarray module across numpy 1.x and 2.x."""
    try:
        from numpy._core import multiarray
    except ImportError:  # numpy < 2
        from numpy.core import multiarray
    return multiarray


# Overridden so that pickles cannot reach the unguarded numpy reconstructors.
# Data written by numpy 1.x names them `numpy.core.*`, numpy 2.x `numpy._core.*`.
_NUMPY_OVERRIDES: Dict[Tuple[str, str], Callable] = {
    ("numpy", "dtype"): _safe_dtype,
    ("numpy", "ndarray"): np.ndarray,
}
for _module in ("numpy.core.multiarray", "numpy._core.multiarray"):
    _NUMPY_OVERRIDES[(_module, "scalar")] = _safe_scalar
    _NUMPY_OVERRIDES[(_module, "_reconstruct")] = _safe_reconstruct

# Non-geobench types that appear in band metadata and task specifications.
_ALLOWED_GLOBALS = frozenset(
    {
        ("affine", "Affine"),
        ("rasterio.crs", "CRS"),
        ("datetime", "date"),
        ("datetime", "datetime"),
        ("datetime", "time"),
        ("datetime", "timedelta"),
        ("datetime", "timezone"),
        ("collections", "OrderedDict"),
    }
)

# Module names geobench pickles have been written under. The `geobench.io` and
# `ccb` spellings are historical and remapped in `geobench/__init__.py`.
_GEOBENCH_MODULES = frozenset(
    {
        "geobench",
        "geobench.dataset",
        "geobench.label",
        "geobench.task",
        "geobench.io",
        "geobench.io.dataset",
        "geobench.io.label",
        "geobench.io.task",
        "ccb",
        "ccb.io",
        "ccb.io.dataset",
        "ccb.io.label",
        "ccb.io.task",
    }
)


class SafeUnpickler(pickle.Unpickler):
    """Unpickler that only reconstructs types written by geobench."""

    def find_class(self, module: str, name: str) -> Any:
        """Resolve a pickled global, refusing anything outside the allowlist.

        Args:
            module: module the pickle refers to
            name: attribute name within that module

        Returns:
            the resolved class or reconstructor

        Raises:
            UnsafePickleError: if the global is not on the allowlist
        """
        key = (module, name)
        if key in _NUMPY_OVERRIDES:
            return _NUMPY_OVERRIDES[key]

        if key in _ALLOWED_GLOBALS:
            return super().find_class(module, name)

        if module in _GEOBENCH_MODULES:
            obj = super().find_class(module, name)
            # Reject anything that is merely imported into a geobench module,
            # and any non-class attribute, e.g. a function or a module.
            if isinstance(obj, type) and obj.__module__ in _GEOBENCH_MODULES:
                return obj
            raise UnsafePickleError(f"{module}.{name} is not a geobench class")

        raise UnsafePickleError(
            f"refusing to unpickle {module}.{name}: it is not part of the geobench "
            "data format. This file was not written by geobench and may be malicious."
        )


def safe_load(file: IO[bytes]) -> Any:
    """Unpickle from a binary file object, restricted to geobench types.

    Args:
        file: binary file object to read from

    Returns:
        the unpickled object
    """
    return SafeUnpickler(file).load()


def safe_loads(data: bytes) -> Any:
    """Unpickle from a bytes object, restricted to geobench types.

    Args:
        data: pickled bytes

    Returns:
        the unpickled object
    """
    return SafeUnpickler(io.BytesIO(data)).load()
