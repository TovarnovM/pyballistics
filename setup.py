# python setup.py bdist_wheel
# pip install -e .

# python setup.py sdist
# $Env:CYTHONIZE=1
# pip install twine
# twine upload dist/*



import os
from setuptools import setup, find_packages, Extension


# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


extensions = [
    Extension("pyballistics.termo", ["src/pyballistics/termo.pyx"], language="c"),
    Extension("pyballistics.lagrange", ["src/pyballistics/lagrange.pyx"], language="c"),
    Extension("pyballistics.termalconduct", ["src/pyballistics/termalconduct.pyx"], language="c"),
]

CYTHONIZE = bool(int(os.getenv("CYTHONIZE", 0)))

if CYTHONIZE:
    from Cython.Build import cythonize
    compiler_directives = {"language_level": 3, "embedsignature": True, "boundscheck": False, "wraparound": False, "cdivision": True, 'nonecheck': False}
    extensions = cythonize(extensions, compiler_directives=compiler_directives, annotate=False)
else:
    extensions = no_cythonize(extensions)

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

with open("requirements-dev.txt") as fp:
    dev_requires = fp.read().strip().split("\n")

setup(
    ext_modules=extensions,
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "docs": ["sphinx", "sphinx-rtd-theme"]
    },
    include_package_data=True
    # data_files = [('', ['src/pyballistics/gpowders_si.csv'])]
)