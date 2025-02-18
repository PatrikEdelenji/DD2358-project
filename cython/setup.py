from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("optimized_funcs.pyx", language_level="3"),
    include_dirs=[np.get_include()]
)