# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt


def DSPF_sp_biquad(x, params, delay):
    a1 = params[4]
    a2 = params[5]
    b0 = params[0]
    b1 = params[1]
    b2 = params[2]
    d0 = delay[0]
    d1 = delay[1]
    r = np.empty([len(x)], dtype=float)
    for i in range(0, len(x)):
        x_i = x[i]
        r[i] = b0 * x_i + d0
        d0 = b1 * x_i - a1 * r[i] + d1
        d1 = b2 * x_i - a2 * r[i]
    delay[0] = d0
    delay[1] = d1
    return r, delay


def display_signal(signal, t, fs):
    plt.figure(2)
    plt.plot(t, signal, drawstyle='steps-post')
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.grid()

    signal_length = len(signal)
    frq = np.arange(signal_length) / np.true_divide(signal_length, fs)
    frq = frq[:int(signal_length / 2)]
    y = np.fft.fft(signal) / signal_length
    y = y[:int(signal_length / 2)]

    plt.figure(3)
    _, _, baseline = plt.stem(frq, abs(y), '-.')
    plt.setp(baseline, 'color', 'r', 'linewidth', 2)
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('|Y(częstotliwość )|')


def main():
    np.set_printoptions(suppress=True)
    fs = 8000  # sampling rate
    order = 14
    fc = 1000/fs
    char = 'lowpass'
    type = 'butter'
    sos = sig.iirfilter(order, fc, btype=char, ftype=type, output='sos')
    z, p, k = sig.iirfilter(order, fc, btype=char, ftype=type, output='zpk')
    sos_ = sos.copy()
    print('k = ', k)
    print(sos)
    for i in range(0, int(order/2)):
        sos_local = sos_[:(i+1), :]
        b, a = sig.sos2tf(sos_local)
        _, h = sig.freqz(b, a)
        k = np.max(np.abs(h))
        print('k{} = {}'.format(i+1, k))
        # dzielimy współczynniki b sekcji  przez k1
        sos_[i, :3] /= k
        print(sos_)

    b, a = sig.sos2tf(sos_)
    _, h = sig.freqz(b, a)
    k = np.max(np.abs(h))
    print('Maksymalne wzmocnienie całego filtru kmax= ', k)

    # Kwantyzacja współczynników do formatu Q1.14
    qsos = np.around(16384 * sos_).astype(np.int16)
    print(qsos)

    print("Wzmocnienie każdej sekcji po kwantyzacji: ")
    for i in range(1, int(order/2)+1):
        bi, ai = sig.sos2tf(qsos[:i, :])
        _, hi = sig.freqz(bi, ai)
        ki = np.max(np.abs(hi))
        print('k{} = {}'.format(i, ki))

    z, p, k = sig.sos2zpk(qsos)
    print("Stabilność - jeżeli moduł kadżdego biegunu jest mniejszy od zera to filtr jest stabilny: \n", np.abs(p))

    w, hn = sig.freqz(*sig.sos2tf(sos))
    w = w * fs / (2 * np.pi) / 1000
    hn = 20 * np.log10(np.abs(hn))
    _, hq = sig.freqz(*sig.sos2tf(qsos))
    hq = 20 * np.log10(np.abs(hq))
    fig, ax = plt.subplots()
    ax.plot(w, hn, color='b', linewidth=2, label='Zmiennoprzecinkowy')
    ax.plot(w, hq, color='r', linewidth=2, label='Stałoprzecinkowy')
    ax.set_xlabel('Częstotliwość [kHz]')
    ax.set_ylabel('Amplituda [dB]')
    ax.legend()

    ts = 1 / fs  # sampling interval
    t = np.arange(0, 1, ts)  # time vector
    ff = 1000  # frequency of the signal
    signal = np.sin(2 * np.pi * ff * t)
    delay = np.array([0.0, 0.0], dtype=float)
    for i in range(0, int(order/2)):
        signal, delay = DSPF_sp_biquad(signal, sos_[i], delay)
    display_signal(signal, t, fs)
    plt.show()

if __name__ == "__main__":
    main()
