import numpy as np
import matplotlib.pyplot as plt


# FIR filter design function
Fs = 1000
ecg = np.loadtxt("Noisy_ECG_1000Hz_39.dat")
N_ecg = len(ecg)
t = np.arange(N_ecg) / Fs


def design_FIR(Fs, stopbands, transition_width=2.0):

    if Fs <= 0 or transition_width <= 0:
        raise ValueError("Invalid sampling frequency or transition width")

    nyquist = Fs / 2

    for low, high in stopbands:
        if low < 0 or high <= low or high > nyquist:
            raise ValueError("Invalid stopband")

    N = int(np.ceil(3.3 * Fs / transition_width))

    if N % 2 == 0:
        N += 1

    L = 1
    while L < 8 * N:
        L *= 2

    H = np.ones(L)

    for k in range(L):

        if k <= L // 2:
            f = k * Fs / L
        else:
            f = (k - L) * Fs / L

        f = abs(f)

        for low, high in stopbands:

            if low <= f <= high:
                H[k] = 0

            elif low > 0 and low - transition_width < f < low:
                H[k] = min(
                    H[k],
                    0.5 * (1 + np.cos(
                        np.pi * (f - low + transition_width)
                        / transition_width
                    ))
                )

            elif high < f < high + transition_width:
                H[k] = min(
                    H[k],
                    0.5 * (1 - np.cos(
                        np.pi * (f - high)
                        / transition_width
                    ))
                )

    h = np.real(np.fft.ifft(H))

    coefficients = np.zeros(N)
    centre = N // 2

    for i in range(N):
        index = i - centre
        if index < 0:
            index += L
        coefficients[i] = h[index]

    coefficients *= np.hamming(N)

    return coefficients



stopbands = [
    (0, 0.5),
    (49, 51),
    (99, 101),
    (149, 151)
]


# Design FIR coefficients

transition_width = 2.0

coefficients = design_FIR(
    Fs,
    stopbands,
    transition_width
)

N = len(coefficients)

print("Number of coefficients:", N)
print("Filter order:", N - 1)
print("Group delay:", (N - 1) // 2, "samples")


# Calculate frequency response for verification

L = 1

while L < 8 * N:
    L *= 2

response = np.fft.fft(coefficients, L)

frequencies = np.arange(L // 2 + 1) * Fs / L

magnitude = np.abs(response[:L // 2 + 1])

magnitude_db = 20 * np.log10(magnitude + 1e-12)


# Normalise the coefficients at 10 Hz

reference_index = int(round(10 * L / Fs))

gain = magnitude[reference_index]

if gain > 1e-12:
    coefficients /= gain


# FIR filtering using an efficient circular buffer

buffer = np.zeros(N)

filtered_ecg = np.zeros(N_ecg)

index = 0

for n in range(N_ecg):

    buffer[index] = ecg[n]

    result = 0.0

    result += np.dot(
        coefficients[:index + 1],
        buffer[index::-1]
    )

    if index < N - 1:
        result += np.dot(
            coefficients[index + 1:],
            buffer[N - 1:index:-1]
        )

    filtered_ecg[n] = result

    index += 1

    if index == N:
        index = 0

fig, axes = plt.subplots(4, 1, figsize=(14, 14))

display_samples = min(N_ecg, 10 * Fs)

axes[0].plot(t[:display_samples], ecg[:display_samples], linewidth=0.8)
axes[0].set_title("Original Noisy ECG")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

axes[1].plot(t[:display_samples], filtered_ecg[:display_samples], linewidth=0.8)
axes[1].set_title("FIR Filtered ECG")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True)

axes[2].plot(np.arange(N), coefficients, linewidth=0.8)
axes[2].set_title("FIR Filter Coefficients")
axes[2].set_xlabel("Coefficient Index")
axes[2].set_ylabel("Amplitude")
axes[2].grid(True)

axes[3].plot(frequencies, magnitude_db, linewidth=0.8)
axes[3].set_title("FIR Frequency Response")
axes[3].set_xlabel("Frequency (Hz)")
axes[3].set_ylabel("Magnitude (dB)")
axes[3].set_xlim(0, Fs / 2)
axes[3].grid(True)

plt.tight_layout()
plt.show()