from mkl_fft import fft as mkl_fft
from mkl_fft import ifft as mkl_ifft
import numpy as np

x = np.random.rand(8).astype(np.complex64)
fft_mkl = np.empty_like(x)

mkl_fft(x, fft_mkl)
fft_numpy = np.fft.fft(x)

print(np.linalg.norm(fft_mkl - fft_numpy))

ifft_mkl = np.empty_like(fft_mkl)
mkl_ifft(fft_mkl, ifft_mkl)

ifft_numpy = np.fft.ifft(fft_numpy)

print(np.linalg.norm(ifft_mkl - ifft_numpy))
print(np.linalg.norm(ifft_mkl - x))