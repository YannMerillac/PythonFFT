from setuptools import Extension, setup
from Cython.Build import cythonize
#import numpy as np

extensions = [
    Extension("mkl_fft", ["mkl_fft.pyx"],
        include_dirs=["/usr/include/mkl"],
        libraries=["mkl_rt"]),
        #library_dirs=[...]),
]
setup(
    name="mkl_fft",
    ext_modules=cythonize(extensions),
)