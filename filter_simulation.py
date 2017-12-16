import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig


def main():
    Fs = 1000  # sampling rate
    Ts = 1 / Fs  # sampling interval
    t = np.arange(0, 1, Ts)  # time vector
    ff = 50  # frequency of the signal
    signal = np.sin(2*np.pi*ff*t)
    signal_length = len(signal)  # length of the signal
    k = np.arange(signal_length)
    T = signal_length / Fs
    frq = k / T  # two sides frequency range
    frq = frq[range(int(signal_length / 2))]  # one side frequency range

    Y = np.fft.fft(signal) / signal_length  # fft computing and normalization
    Y = Y[:int(signal_length / 2)]

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(t, signal, drawstyle='steps-pre')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(frq, abs(Y), 'r')  # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    plt.show()
    # markerline, stemlines, baseline = plt.stem(frq, abs(Y), '-.')
    # plt.setp(baseline, 'color', 'r', 'linewidth', 2)
    # plt.grid()
    # plt.show()

if __name__ == "__main__":
    main()