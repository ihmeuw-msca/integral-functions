import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "integral_functions.vectorized_funcs",
        ["src/integral_functions/vectorized_funcs.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="integral_functions",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": 3, "profile": False},
    ),
)
