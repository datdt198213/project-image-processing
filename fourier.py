# see DFT https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.02-Discrete-Fourier-Transform.html
# see FFT https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html
from math import ceil, log

import numpy as np


def omega(p, q):
    return np.exp((-2.0 * np.pi * 1j * q) / p)


def fft(x):
    n = len(x)
    if n == 1:
        return x
    x_even, x_odd = fft(x[::2]), fft(x[1::2])
    if np.ndim(x) == 2:
        combined = np.zeros_like(x, dtype=complex)
    if np.ndim(x) == 1:
        combined = np.zeros(n, dtype=complex)

    for m in range(int(n / 2)):
        combined[m] = x_even[m] + omega(n, m) * x_odd[m]
        combined[m + int(n / 2)] = x_even[m] - omega(n, m) * x_odd[m]

    return combined


def ifft(X):
    x = fft(X.conjugate()) / len(X)
    return x


def pad2(x):
    m, n = np.shape(x)
    _M, _N = 2 ** int(ceil(log(m, 2))), 2 ** int(ceil(log(n, 2)))
    _F = np.zeros((_M, _N), dtype=x.dtype)
    _F[0:m, 0:n] = x
    return _F, m, n


def fft2(image):
    origin_m, origin_n = image.shape
    x, m, n = pad2(image)
    fft_x = fft(x)
    fft_y = fft(np.transpose(fft_x))
    result = np.transpose(fft_y)
    return result[:origin_m, :origin_n]


def ifft2(F):
    origin_m, origin_n = F.shape
    x, m, n = pad2(F)
    reverse_fft = fft2(np.conj(F)) / (m * n)
    return reverse_fft[:origin_m, :origin_n]


def fftshift(F):
    m, n = F.shape
    half_m, half_n = int(m / 2), int(n / 2)
    r1, r2 = F[:half_m, :half_n], F[half_m:, :half_n]
    r3, r4 = F[:half_m, half_n:], F[half_m:, half_n:]

    shift_f = np.zeros_like(F)
    # swap r1 <-> r4, r2 <-> r3
    shift_f[half_m:, half_n:], shift_f[:half_m, half_n:] = r1, r2
    shift_f[half_m:, :half_n], shift_f[:half_m, :half_n] = r3, r4
    return shift_f
