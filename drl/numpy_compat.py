"""
Numpy version compatibility shim for model deserialization.

Models serialized with numpy >= 2.0 reference ``numpy._core`` submodules
(e.g. ``numpy._core.numeric``, ``numpy._core.multiarray``).  When loaded
under numpy < 2.0, these modules do not exist and unpickling raises
``ModuleNotFoundError: No module named 'numpy._core.numeric'``.

Calling :func:`patch_numpy_core_compat` before any pickle-load adds
transparent redirects so the unpickler finds the equivalent numpy 1.x
modules under ``numpy.core``.

Safe to call multiple times -- subsequent calls are no-ops.
"""

import importlib
import sys

_PATCHED = False

# Submodules that numpy 2.x exposes under ``numpy._core`` and that
# appear in pickled SB3 / torch artifacts.
_SUBMODULES = [
    "numeric",
    "multiarray",
    "umath",
    "fromnumeric",
    "_methods",
    "_internal",
    "arrayprint",
    "defchararray",
    "function_base",
    "getlimits",
    "shape_base",
    "einsumfunc",
    "overrides",
    "records",
]


def patch_numpy_core_compat() -> None:
    """Ensure ``numpy._core*`` resolves to ``numpy.core*`` if missing.

    Under numpy >= 2.0, ``numpy._core`` already exists -- this function
    detects that and exits immediately.
    """
    global _PATCHED
    if _PATCHED:
        return

    # If numpy._core already exists (numpy >= 2.0), nothing to do.
    try:
        importlib.import_module("numpy._core")
        _PATCHED = True
        return
    except ImportError:
        pass

    import numpy.core  # noqa: F401 -- must exist for the redirect to work

    # Redirect the top-level _core to core
    sys.modules.setdefault("numpy._core", sys.modules["numpy.core"])

    for sub in _SUBMODULES:
        target = f"numpy.core.{sub}"
        alias = f"numpy._core.{sub}"
        try:
            importlib.import_module(target)
            sys.modules.setdefault(alias, sys.modules[target])
        except ImportError:
            # Not every submodule exists on every numpy build; skip silently.
            pass

    _PATCHED = True
