from __future__ import annotations

from .options import (
    get_full_options,
    get_db_powder,
    get_powder_names,
    get_options_sample,
    get_options_agard,
    get_options_sample_2,
)
from .termo import ozvb_termo
from .lagrange import ozvb_lagrange

__all__ = [
    "ozvb_termo",
    "ozvb_lagrange",
    "get_full_options",
    "get_db_powder",
    "get_powder_names",
    "get_options_sample",
    "get_options_agard",
    "get_options_sample_2",
]

try:
    from importlib.metadata import version
    __version__ = version("pyballistics")
except Exception:  # pragma: no cover
    __version__ = "0+unknown"
