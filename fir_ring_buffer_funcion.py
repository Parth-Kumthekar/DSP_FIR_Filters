"""
FIR filter design using IFFT + Kaiser window, with streaming ring-buffer
implementation.

Requirements:
    pip install numpy matplotlib

Optional input:
    Place Noisy_ECG_1000Hz_39.dat in the same directory as this script.

The script:
  1. Designs a multi-band FIR band-stop filter from sampling rate and stopbands.
  2. Estimates the tap count from transition bandwidth and attenuation.
  3. Builds the desired frequency response, applies IFFT, truncates, windows,
     and normalises the coefficients.
  4. Filters samples using a circular/ring buffer.
  5. Plots the input/output signals and the filter frequency response.

Important:
  - This is an educational IFFT-based design. Always verify measured
    passband ripple and stopband attenuation for your application.
  - A causal FIR filter has group delay (num_taps - 1) / (2 * fs)
    when its coefficients are symmetric.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _merge_bands(stopbands):
    """Validate, sort, and merge overlapping/touching stopbands."""
    if stopbands is None:
        raise ValueError("stopbands must be a list of (low_hz, high_hz) pairs")

    bands = []
    for band in stopbands:
        if len(band) != 2:
            raise ValueError(f"Each stopband must have 2 values: {band}")
        low, high = map(float, band)
        bands.append((low, high))

    bands.sort()
    merged = []
    for low, high in bands:
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


def design_fir_ifft(fs, stopbands, transition_hz=5.0,
                    attenuation_db=60.0):
    """
    Design a real, symmetric, odd-length FIR multi-band-stop filter.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    stopbands : sequence of (low_hz, high_hz)
        Frequency ranges to attenuate. Use (0, f_high) for DC removal.
    transition_hz : float
        Width of each cosine transition region, in Hz.
    attenuation_db : float
        Target Kaiser-window stopband attenuation estimate, in dB.

    Returns
    -------
    h : ndarray
        FIR coefficients, newest-sample coefficient first in the
        ring-buffer convolution implementation.
    info : dict
        Design parameters and estimated filter length.
    """
    fs = float(fs)
    transition_hz = float(transition_hz)
    attenuation_db = float(attenuation_db)

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be a positive finite number")

    nyquist = fs / 2.0
    if not np.isfinite(transition_hz) or not (0 < transition_hz < nyquist):
        raise ValueError("transition_hz must be between 0 and Nyquist")
    if not np.isfinite(attenuation_db) or attenuation_db < 21:
        raise ValueError("attenuation_db must be >= 21 dB for this Kaiser estimate")

    bands = _merge_bands(stopbands)
    if not bands:
        return np.array([1.0]), {
            "order": 0, "num_taps": 1, "ifft_size": 1,
            "transition_hz": transition_hz,
            "attenuation_db": attenuation_db,
            "kaiser_beta": 0.0,
            "stopbands": []
        }

    for low, high in bands:
        if not (0 <= low < high <= nyquist):
            raise ValueError(
                f"Invalid stopband {(low, high)}. Require 0 <= low < high <= {nyquist} Hz."
            )
        # A full transition must fit on each non-DC/non-Nyquist edge.
        if low > 0 and low < transition_hz:
            raise ValueError(
                f"Stopband lower edge {low} Hz is too close to DC for "
                f"a {transition_hz} Hz left transition."
            )
        if high < nyquist and high + transition_hz > nyquist:
            raise ValueError(
                f"Stopband upper edge {high} Hz leaves insufficient room "
                f"for a {transition_hz} Hz right transition before Nyquist."
            )

    # Kaiser order estimate. Delta omega is in radians/sample.
    delta_omega = 2.0 * np.pi * transition_hz / fs
    order = int(np.ceil(
        (attenuation_db - 8.0) / (2.285 * delta_omega)
    ))

    # Even order gives odd number of taps (Type-I linear phase).
    if order % 2:
        order += 1
    num_taps = order + 1

    # Kaiser beta approximation.
    A = attenuation_db
    if A > 50:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21:
        beta = (0.5842 * (A - 21.0) ** 0.4
                + 0.07886 * (A - 21.0))
    else:
        beta = 0.0

    # Dense frequency grid for the IFFT design. This is not the tap count.
    required_size = max(
        8 * num_taps,
        int(np.ceil(8.0 * fs / transition_hz))
    )
    nfft = 1 << int(np.ceil(np.log2(required_size)))

    # fftfreq returns positive and negative frequencies.
    # The absolute value makes the desired response even (conjugate symmetric).
    freq = np.abs(np.fft.fftfreq(nfft, d=1.0 / fs))
    H = np.ones(nfft, dtype=float)

    # Build each band's response and combine them.
    for low, high in bands:
        response = np.ones(nfft, dtype=float)

        if low == 0:
            # DC stopband: there is no transition below zero.
            response[freq <= high] = 0.0
        else:
            left = (freq >= low - transition_hz) & (freq < low)
            response[left] = 0.5 * (
                1.0 + np.cos(
                    np.pi * (freq[left] - low + transition_hz)
                    / transition_hz
                )
            )
            response[(freq >= low) & (freq <= high)] = 0.0

        if high == nyquist:
            # No right transition beyond Nyquist.
            response[freq >= high] = 0.0
        else:
            right = (freq > high) & (freq <= high + transition_hz)
            response[right] = 0.5 * (
                1.0 - np.cos(
                    np.pi * (freq[right] - high) / transition_hz
                )
            )

        H = np.minimum(H, response)

    # IFFT: frequency response -> periodic impulse response.
    # fftshift places the zero-lag sample at the centre for symmetric extraction.
    impulse_centered = np.fft.fftshift(np.fft.ifft(H).real)

    centre = nfft // 2
    half = num_taps // 2
    h_ideal_truncated = impulse_centered[centre-half:centre+half+1]

    # Apply symmetric Kaiser window to reduce truncation sidelobes.
    window = np.kaiser(num_taps, beta)
    h = h_ideal_truncated * window

    # Find the widest available passband and normalise gain at its midpoint.
    expanded = []
    for low, high in bands:
        expanded.append((max(0.0, low - (0 if low == 0 else transition_hz)),
                         min(nyquist, high + (0 if high == nyquist else transition_hz))))
    expanded.sort()

    merged_expanded = []
    for low, high in expanded:
        if merged_expanded and low <= merged_expanded[-1][1]:
            merged_expanded[-1] = (
                merged_expanded[-1][0],
                max(merged_expanded[-1][1], high)
            )
        else:
            merged_expanded.append((low, high))

    passbands = []
    cursor = 0.0
    for low, high in merged_expanded:
        if low > cursor:
            passbands.append((cursor, low))
        cursor = max(cursor, high)
    if cursor < nyquist:
        passbands.append((cursor, nyquist))

    if not passbands:
        raise ValueError("The specified stopbands leave no passband for gain normalisation.")

    reference_band = max(passbands, key=lambda item: item[1] - item[0])
    f_ref = 0.5 * (reference_band[0] + reference_band[1])

    n = np.arange(num_taps)
    reference_gain = abs(np.sum(
        h * np.exp(-2j * np.pi * f_ref * n / fs)
    ))
    if reference_gain < 1e-12:
        raise ValueError("Passband reference gain is too small to normalise.")
    h = h / reference_gain

    info = {
        "order": order,
        "num_taps": num_taps,
        "ifft_size": nfft,
        "transition_hz": transition_hz,
        "attenuation_db": attenuation_db,
        "kaiser_beta": beta,
        "normalisation_frequency_hz": f_ref,
        "stopbands": bands
    }
    return h, info


class FIRRingBuffer:
    """Streaming FIR filter using a circular buffer."""

    def __init__(self, coefficients):
        self.h = np.asarray(coefficients, dtype=float)
        if self.h.ndim != 1 or self.h.size == 0:
            raise ValueError("coefficients must be a non-empty 1-D array")
        self.num_taps = self.h.size
        self.buffer = np.zeros(self.num_taps, dtype=float)
        self.write_ptr = 0

    def reset(self):
        """Clear stored samples and reset the write pointer."""
        self.buffer.fill(0.0)
        self.write_ptr = 0

    def process_sample(self, sample):
        """Process one new sample and return one output sample."""
        self.buffer[self.write_ptr] = float(sample)

        acc = 0.0
        read_ptr = self.write_ptr

        # h[0] multiplies newest sample, h[1] the previous sample, etc.
        for k in range(self.num_taps):
            acc += self.h[k] * self.buffer[read_ptr]
            read_ptr -= 1
            if read_ptr < 0:
                read_ptr = self.num_taps - 1

        self.write_ptr += 1
        if self.write_ptr == self.num_taps:
            self.write_ptr = 0

        return acc

    def process_block(self, samples):
        """Process an array/block while retaining state between calls."""
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 1:
            raise ValueError("samples must be a 1-D array")
        output = np.empty_like(samples, dtype=float)
        for i, sample in enumerate(samples):
            output[i] = self.process_sample(sample)
        return output


def make_demo_signal(fs, duration=5.0):
    """Create an ECG-like demonstration signal if no data file is present."""
    t = np.arange(int(fs * duration)) / fs
    # Simple pulse-like ECG approximation plus baseline, mains and high-frequency noise.
    heart_rate_hz = 1.2
    phase = (t * heart_rate_hz) % 1.0
    p = 0.12 * np.exp(-0.5 * ((phase - 0.18) / 0.035) ** 2)
    q = -0.15 * np.exp(-0.5 * ((phase - 0.30) / 0.012) ** 2)
    r = 1.00 * np.exp(-0.5 * ((phase - 0.32) / 0.010) ** 2)
    s = -0.25 * np.exp(-0.5 * ((phase - 0.35) / 0.015) ** 2)
    tw = 0.30 * np.exp(-0.5 * ((phase - 0.58) / 0.07) ** 2)
    clean = p + q + r + s + tw
    noisy = (clean
             + 0.15 * np.sin(2 * np.pi * 0.5 * t)
             + 0.12 * np.sin(2 * np.pi * 50 * t)
             + 0.05 * np.sin(2 * np.pi * 160 * t))
    return t, noisy


def plot_results(x, y, fs, h):
    """Plot time-domain signals and linear-axis frequency response."""
    t = np.arange(len(x)) / fs
    nfft = max(8192, 2 ** int(np.ceil(np.log2(8 * len(h)))))
    H = np.fft.rfft(h, nfft)
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    H_db = 20 * np.log10(np.maximum(np.abs(H), 1e-12))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(t, x, linewidth=0.8)
    axes[0].set_title("Original input signal")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    axes[1].plot(t, y, linewidth=0.8)
    axes[1].set_title("FIR output (ring-buffer implementation)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    axes[2].plot(f, H_db, linewidth=1.0)
    axes[2].set_title("FIR magnitude response (linear frequency axis)")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Magnitude (dB)")
    axes[2].set_xlim(0, fs / 2)
    axes[2].set_ylim(-100, 5)
    axes[2].grid(True)
    

    plt.tight_layout()
    plt.show()


def main():
    fs = 1000.0

    # Edit these ranges to select the frequencies to remove.
    # (0, 2) removes DC and very-low-frequency content.
    stopbands = [
        (0, 2),
        (48, 52),
        (155, 165)
    ]

    transition_hz = 5.0
    attenuation_db = 60.0

    h, info = design_fir_ifft(
        fs=fs,
        stopbands=stopbands,
        transition_hz=transition_hz,
        attenuation_db=attenuation_db
    )

    print("FIR design information")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print(f"\nNumber of coefficients: {len(h)}")
    print(f"Filter order: {len(h) - 1}")
    print(f"Group delay: {(len(h) - 1) / (2 * fs):.6f} seconds")

    data_file = Path(__file__).with_name("Noisy_ECG_1000Hz_39.dat")

    if data_file.exists():
        x = np.loadtxt(data_file, dtype=float)
        x = np.asarray(x).squeeze()
        if x.ndim != 1:
            raise ValueError("The ECG data file must contain one signal column.")
        print(f"\nLoaded ECG data: {data_file}")
    else:
        print(f"\n'{data_file.name}' not found.")
        print("Using a synthetic ECG-like signal for demonstration.")
        _, x = make_demo_signal(fs)

    fir = FIRRingBuffer(h)
    y = fir.process_block(x)

    plot_results(x, y, fs, h)


if __name__ == "__main__":
    main()
