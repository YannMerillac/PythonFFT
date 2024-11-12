from libcpp.vector cimport vector
from libcpp.complex cimport complex
import cython
#cimport numpy as cnp
#cnp.import_array()

ctypedef double complex complex128_t
ctypedef float complex complex64_t

ctypedef vector[ptrdiff_t] stride_t
ctypedef vector[size_t] shape_t    

cdef extern from "pocketfft/pocketfft_hdronly.h" namespace "pocketfft":

    void c2c(const shape_t &shape, const stride_t &stride_in,
        const stride_t &stride_out, const shape_t &axes, bint forward,
        const complex[float] *data_in, complex[float] *data_out, float fct,
        size_t nthreads) nogil
    
@cython.wraparound(False)
def fft(complex64_t[:] x, complex64_t[:] y, size_t n_threads=1):
    cdef size_t i, ii
    cdef size_t ndata=x.shape[0]
    cdef shape_t shape = shape_t(1)
    shape[0] = ndata
    cdef stride_t strided = stride_t(1)
    cdef size_t tmpd=sizeof(complex[float])
    strided[0] = tmpd

    cdef shape_t axes
    axes.push_back(0)

    cdef complex[float] * x_ptr = <complex[float]*>(&x[0])
    cdef complex[float] * y_ptr = <complex[float]*>(&y[0])
  
    cdef float fct = 1
    with nogil:
        c2c(shape, strided, strided, axes, 1, x_ptr, y_ptr, fct, n_threads)
