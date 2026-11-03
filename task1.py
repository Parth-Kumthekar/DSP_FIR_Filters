import numpy as np
import matplotlib.pyplot as plt

# Load data
ecg = np.loadtxt("Noisy_ECG_1000Hz_39.dat")
N = len(ecg)
fs = 1000
t = np.arange(N) / fs

# Custom frequency axis function using only np.fft.fft
def freq_axis(n, fs):
    k = np.arange(n)
    return np.minimum(k, n - k) * fs / n

f = freq_axis(N, fs)
X = np.fft.fft(ecg)
magnitude = np.abs(X) / N
magnitude[1:-1] *= 2

# Plotting: 3 subplots (Time, Full Frequency, Zoomed Frequency 0-100Hz)
fig, axes = plt.subplots(3, 1, figsize=(10, 9))

# 1. Time Plot (zoomed in to 0 - 0.5s)
axes[0].plot(t, ecg, lw=0.8)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Time Domain Plot (Zoomed: 0 - 0.5s)")
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 0.5)

# 2. Frequency Spectrum Plot (Full)
half = N // 2
axes[1].plot(f[1:half], magnitude[1:half], lw=0.8)
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Magnitude")
axes[1].set_title("Frequency Spectrum Plot (Full)")
axes[1].set_xlim(0, fs / 2)
axes[1].grid(True, alpha=0.3)

# 3. Zoomed Frequency Spectrum Plot (0 - 100 Hz)
axes[2].plot(f[1:half], magnitude[1:half], lw=0.8)
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("Magnitude")
axes[2].set_title("Frequency Spectrum Plot (Zoomed: 0 - 100 Hz)")
axes[2].set_xlim(0, 100)
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
plt.show()