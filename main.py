import numpy as np
import math
import time
import warnings
import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def fourier_coefficient(x, k):
    A_k = 0
    B_k = 0
    for n in range(N):
        A_k += x[n] * np.cos(2 * np.pi * k * n / N)
        B_k += x[n] * np.sin(2 * np.pi * k * n / N)
    c_k = complex(A_k, -B_k) / N
    add = N
    mul = N * 3
    return c_k, add, mul


def find_ck(x, N):
    oper_add = 0
    oper_mul = 0
    Ck = np.zeros(N, dtype=complex)
    for k in range(N):
        Ck[k], num_add, num_mul = fourier_coefficient(x, k)
        oper_add += num_add
        oper_mul += num_mul
        print(f"C_{k} = {Ck[k]}")
    print("Number of addition:", oper_add)
    print("Number of multiplication:", oper_mul)
    print("Number of operation:", oper_add + oper_mul)
    return Ck


def fft(x):
    global add, mul
    add += len(x) - 1
    mul += len(x) // 2
    n = len(x)
    if n == 1:
        return x
    else:
        even = fft(x[0::2])
        odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(n) / n)
        return np.concatenate([even + factor[:n // 2] * odd, even + factor[n // 2:] * odd])


N = 31
x = np.random.random(N)

start_time_dft = time.time()
C = find_ck(x, N)
end_time_dft = time.time()
print("Time for DFT: ", end_time_dft - start_time_dft)

M = 2**round(math.log2(N))
print(round(math.log2(N)))
x = np.concatenate([x, np.zeros(abs(M - N))])
print(x)

start_time_fft = time.time()
add = 0
mul = 0
FFT = fft(x)
print("Number of addition:", add)
print("Number of multiplication:", mul)
print("Number of operation:", add + mul)
end_time_fft = time.time()
print("Time for FFT: ", end_time_fft - start_time_fft)

amp_dft = abs(C)
phase_dft = np.angle(C)
ampl_fft = abs(np.array(FFT))
phase_fft = np.angle(FFT)

# plt.stem(amp_dft)
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')
# plt.title('Amplitude chart')
# plt.show()
#
# plt.stem(phase_dft)
# plt.xlabel('Frequency')
# plt.ylabel('Phase')
# plt.title('Phase graph')
# plt.show()

plt.stem(ampl_fft)
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Amplitude chart')
plt.show()

plt.stem(phase_fft)
plt.xlabel('Frequency')
plt.ylabel('Phase')
plt.title('Phase graph')
plt.show()