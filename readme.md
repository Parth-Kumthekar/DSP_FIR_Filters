# ECG FIR Filter — Task 2

## Files

- `ecg_fir_task2.py` — designs an FIR filter, applies it sample by sample, and displays the plots.
- `Noisy_ECG_1000Hz_39.dat` — the ECG data file expected by the script. Keep it in the same folder as the Python file.

## Requirements

Install Python 3 with NumPy and Matplotlib:

```bash
python -m pip install numpy matplotlib
```

## Run

Place `ecg_fir_task2.py` and `Noisy_ECG_1000Hz_39.dat` in the same directory, open a terminal in that directory, and run:

```bash
python ecg_fir_task2.py
```

The script uses a relative filename, not an absolute path. If the data file has a different name, change the `filename` value near the start of `main()`.

## Filter design

The sampling rate is set to 1000 Hz, as indicated by the supplied data filename. The stopbands are set to:

- 0–0.5 Hz to reduce DC and very slow baseline components.
- 49–51 Hz to attenuate 50 Hz mains interference.
- 99–101 Hz to attenuate the second harmonic.
- 149–151 Hz to attenuate the third harmonic.

The transition width is set to 2 Hz. The approximate Hamming-window tap-count rule is:

`N = ceil(3.3 * Fs / transition_width)`

The tap count is made odd so that the FIR has an integer centre. A frequency-sampling response is built symmetrically, transformed using the inverse FFT, centred, truncated to the chosen tap count, and multiplied by a Hamming window. The coefficients are scaled to give unity magnitude at 10 Hz.

## Filtering implementation

The ECG is filtered one sample at a time. A circular delay line stores previous input samples, and each output is calculated from the FIR coefficients and the stored samples. The delay line is not shifted for each new sample.

The FFT is used only to create the FIR coefficients and to display the filter's frequency response. It is not used to filter the ECG data.

## Plots

The script displays one figure containing four vertically stacked plots:

1. Original ECG.
2. Sample-by-sample FIR-filtered ECG.
3. FIR coefficients.
4. FIR magnitude response.

The first two plots show up to the first 10 seconds to keep the time-domain plots readable.

## Notes

- The sampling rate and stopband edges should be checked against the assignment data/specification before submission.
- This is a causal FIR implementation. At the start of the recording, the delay line is initially zero, so the first outputs include the normal filter start-up transient.
- The tap-count formula is an estimate; verify the achieved stopband attenuation and transition widths from the plotted response.
