from libc.stdlib cimport malloc, free
import cython
#cimport numpy as cnp
#cnp.import_array()

cdef extern from "mkl.h":
    cdef struct _MKL_Complex8:
        float real
        float imag
    ctypedef _MKL_Complex8 MKL_Complex8
    ctypedef long int MKL_LONG

    cdef struct DFTI_DESCRIPTOR:
        pass
    ctypedef DFTI_DESCRIPTOR *DFTI_DESCRIPTOR_HANDLE

    cdef enum DFTI_CONFIG_VALUE:
        DFTI_SINGLE
        DFTI_COMPLEX
        DFTI_NOT_INPLACE

    cdef enum DFTI_CONFIG_PARAM:
        DFTI_PLACEMENT
        DFTI_BACKWARD_SCALE

    MKL_LONG DftiCreateDescriptor(DFTI_DESCRIPTOR_HANDLE*,
                                  DFTI_CONFIG_VALUE,
                                  DFTI_CONFIG_VALUE,
                                  MKL_LONG, ...)

    MKL_LONG DftiSetValue(DFTI_DESCRIPTOR_HANDLE, DFTI_CONFIG_PARAM, ...)
    MKL_LONG DftiCommitDescriptor(DFTI_DESCRIPTOR_HANDLE)
    MKL_LONG DftiComputeForward(DFTI_DESCRIPTOR_HANDLE, void*, ...)
    MKL_LONG DftiFreeDescriptor(DFTI_DESCRIPTOR_HANDLE*)
    MKL_LONG DftiComputeBackward(DFTI_DESCRIPTOR_HANDLE, void*, ...)


ctypedef double complex complex128_t
ctypedef float complex complex64_t

@cython.boundscheck(False)
@cython.wraparound(False)
def fft(complex64_t[:] x, complex64_t[:] y):
    cdef int i
    cdef int n = x.shape[0]
    cdef DFTI_DESCRIPTOR_HANDLE handle
    #cdef MKL_Complex8* y = <MKL_Complex8*> malloc(n * sizeof(MKL_Complex8))
    DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_COMPLEX, 1, n)
    DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)
    DftiCommitDescriptor(handle)
    cdef void* x_ptr = &x[0]
    cdef void* y_ptr = &y[0]
    DftiComputeForward(handle, x_ptr, y_ptr)
    DftiFreeDescriptor(&handle)


@cython.boundscheck(False)
@cython.wraparound(False)
def ifft(complex64_t[:] x, complex64_t[:] y):
    cdef int i
    cdef int n = x.shape[0]
    cdef DFTI_DESCRIPTOR_HANDLE handle
    #cdef MKL_Complex8* y = <MKL_Complex8*> malloc(n * sizeof(MKL_Complex8))
    DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_COMPLEX, 1, n)
    DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)
    DftiSetValue(handle, DFTI_BACKWARD_SCALE, 1.0/n)
    DftiCommitDescriptor(handle)
    cdef void* x_ptr = &x[0]
    cdef void* y_ptr = &y[0]
    DftiComputeBackward(handle, x_ptr, y_ptr)
    DftiFreeDescriptor(&handle)


cdef class FFTHandler(object):
    cdef DFTI_DESCRIPTOR_HANDLE handle
    cdef public int N
    
    def __cinit__(self, N):
        self.N = N
        DftiCreateDescriptor(&self.handle, DFTI_SINGLE, DFTI_COMPLEX, 1, self.N)
        
    def __dealloc__(self):
        DftiFreeDescriptor(&self.handle)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fft(self, complex64_t[:] x, complex64_t[:] y):
        DftiSetValue(self.handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)
        DftiCommitDescriptor(self.handle)
        cdef void* x_ptr = &x[0]
        cdef void* y_ptr = &y[0]
        DftiComputeForward(self.handle, x_ptr, y_ptr)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def ifft(self, complex64_t[:] x, complex64_t[:] y):
        DftiSetValue(self.handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE)
        DftiSetValue(self.handle, DFTI_BACKWARD_SCALE, 1.0/self.N)
        DftiCommitDescriptor(self.handle)
        cdef void* x_ptr = &x[0]
        cdef void* y_ptr = &y[0]
        DftiComputeBackward(self.handle, x_ptr, y_ptr)