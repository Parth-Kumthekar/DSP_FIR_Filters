
import numpy as np
import matplotlib.pyplot as plt



def design_FIR(Fs, stopbands, transition_width=1.0):

    # Check that the sampling frequency and transition width
    # are valid.
    if Fs <= 0 or transition_width <= 0:
        raise ValueError("Sampling frequency and transition width must be positive.")

    nyquist = Fs / 2.0

    # Check that all stopband frequency ranges are valid.
    for band in stopbands:

        low = band[0]
        high = band[1]

        if low < 0 or high <= low or high > nyquist:
            raise ValueError("Invalid stopband frequency range.")


  

    N = int(np.ceil(3.3 * Fs / transition_width))

    if N % 2 == 0:
        N += 1


    filter_order = N - 1
    group_delay = filter_order / 2

    print("Number of coefficients:", N)
    print("Filter order:", filter_order)
    print("Group delay:", group_delay, "samples")



    # Use an IFFT length significantly larger than the
    # number of FIR coefficients to obtain a fine frequency grid.

    L = 1

    while L < 8 * N:
        L *= 2


  
    H = np.ones(L)


    for k in range(L):

        # Manually calculate the frequency corresponding to bin k.

        if k <= L // 2:
            frequency = k * Fs / L
        else:
            frequency = (k - L) * Fs / L

        frequency = abs(frequency)

        # Examine each specified stopband.

        for band in stopbands:

            low = band[0]
            high = band[1]

            response = 1.0

            # Set the required stopband frequencies to zero.

            if low <= frequency <= high:

                response = 0.0

            # Create a raised-cosine transition before the
            # lower edge of a stopband.
            # This is not needed when removing DC.

            elif low > 0 and low - transition_width < frequency < low:

                position = (frequency - (low - transition_width)) / transition_width

                response = 0.5 * (1.0 + np.cos(np.pi * position))

            # Create a raised-cosine transition after the
            # upper edge of a stopband.

            elif high < frequency < high + transition_width:

                position = (frequency - high) / transition_width

                response = 0.5 * (1.0 - np.cos(np.pi * position))

            # If stopbands overlap, retain the lowest response.

            H[k] = min(H[k], response)


    impulse_response = np.real(np.fft.ifft(H))

    # Centre the impulse response manually.
    # This creates a symmetric, linear-phase FIR filter.

    coefficients = np.zeros(N)

    centre = N // 2

    for i in range(N):

        index = i - centre

        if index < 0:
            index = L + index

        coefficients[i] = impulse_response[index]


   
    # A Hamming window reduces unwanted ripples caused by
    # truncating the impulse response.

    window = np.hamming(N)

    for i in range(N):

        coefficients[i] *= window[i]


    # Return the calculated FIR coefficients.

    return coefficients



Fs = 1000


stopbands = [

    (0, 0.5),

    (49, 51),

    (99, 101),

    (149, 151)

]



coefficients = design_FIR(
    Fs,
    stopbands,
    transition_width=1.0
)

print("\nFIR coefficients successfully calculated.")


plt.figure(figsize=(12, 4))

plt.plot(coefficients)

plt.title("FIR Filter Impulse Response")
plt.xlabel("Coefficient Number")
plt.ylabel("Amplitude")

plt.grid()
plt.tight_layout()
plt.show()



frequency_response = np.fft.fft(coefficients, L)

magnitude = np.abs(frequency_response)

frequencies = np.arange(L // 2 + 1) * Fs / L

magnitude_db = 20 * np.log10(
    magnitude[:L // 2 + 1] + 1e-12
)

# Plot the positive frequency response.

plt.figure(figsize=(12, 5))

plt.plot(frequencies, magnitude_db)

plt.title("Designed FIR Filter Frequency Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")

plt.xlim(0, Fs / 2)
plt.ylim(-100, 10)

plt.grid()
plt.tight_layout()
plt.show()