# setup.py (нужен только для Cython ext_modules)
from __future__ import annotations

import os
from setuptools import Extension, setup
from Cython.Build import cythonize


def _ext_language() -> str:
    """
    Управляем генерацией .c/.cpp явно:
      - PYBALLISTICS_CYTHON_LANG=c   -> language="c"   -> Cython генерирует .c
      - PYBALLISTICS_CYTHON_LANG=cpp -> language="c++" -> Cython генерирует .cpp
    """
    lang = os.getenv("PYBALLISTICS_CYTHON_LANG", "c").strip().lower()
    if lang in ("cpp", "c++"):
        return "c++"
    return "c"


LANG = _ext_language()

extra_compile_args = ["-O3"]
if LANG == "c++":
    extra_compile_args += ["-std=c++17"]

extensions = [
    Extension(
        "pyballistics.termo",
        ["src/pyballistics/termo.pyx"],
        language=LANG,
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "pyballistics.lagrange",
        ["src/pyballistics/lagrange.pyx"],
        language=LANG,
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "pyballistics.termalconduct",
        ["src/pyballistics/termalconduct.pyx"],
        language=LANG,
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
    )
)
