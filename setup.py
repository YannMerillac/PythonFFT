from setuptools import Extension, setup
from Cython.Build import cythonize
#import numpy as np

extensions = [
    Extension("mkl_fft", ["mkl_fft.pyx"],
        include_dirs=["/usr/include/mkl"],
        libraries=["mkl_rt"],),
        #extra_compile_args=["-O3", "-ffast-math", "-ftree-vectorize"],)
        #library_dirs=[...]),
        
    Extension("pocketfft", ["pocketfft.pyx"],
        include_dirs=["pocketfft"],
        language="c++",
        extra_compile_args=["-O3", "-ffast-math", "-DPOCKETFFT_PTHREADS", "-DPOCKETFFT_CACHE_SIZE=16"],
        libraries=[],)
        #extra_compile_args=["-O3", "-ffast-math", "-ftree-vectorize"],)
        #library_dirs=[...]),
]
setup(
    name="mkl_fft",
    ext_modules=cythonize(extensions),
)