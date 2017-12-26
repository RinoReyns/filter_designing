# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt


def get_value(display_string):
    try:
        value = float(input(display_string))
    except ValueError:
        value = 0.0
    return value


def process_signal_params(display):
    signal_frequency = get_value(display)
    while signal_frequency < 0.001:
        print("Wartość jest nieprawidłowa.")
        signal_frequency = get_value(display)
    return signal_frequency


def get_signal_params():
    signal_frequency = process_signal_params("Podaj częstotliwość sygnału w Hz: ")
    amplitude = process_signal_params("Podaj amplitudę sygnału: ")
    return signal_frequency, amplitude


def process_sample_rate(signal_frequency):
    fs = get_value("Podaj czestotliwość próbkowania sygnału [Hz]: ")
    while fs < 4 * signal_frequency:
        print("Czestotliwość próbkowania jest nieprawidłowa.")
        fs = get_value("Podaj czestotliwość próbkowania sygnału [Hz]: ")
    ts = 1 / fs
    return fs, ts


def DSPF_sp_biquad(x, params, delay):
    a1 = params[4]
    a2 = params[5]
    b0 = params[0]
    b1 = params[1]
    b2 = params[2]
    d0 = delay[0]
    d1 = delay[1]
    r = np.empty([len(x)], dtype=np.float16)
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


def filter_designing(fs):
    filter_order = 14
    fc = 996 / fs
    char = 'lowpass'
    type = 'butter'
    sos = sig.iirfilter(filter_order, fc, btype=char, ftype=type, output='sos')
    z, p, k = sig.iirfilter(filter_order, fc, btype=char, ftype=type, output='zpk')
    sos_ = sos.copy()
    print('k = ', k)
    print(sos)
    for i in range(0, int(filter_order / 2)):
        sos_local = sos_[:(i + 1), :]
        b, a = sig.sos2tf(sos_local)
        _, h = sig.freqz(b, a)
        k = np.max(np.abs(h))
        print('k{} = {}'.format(i + 1, k))
        # dzielnie współczynników b sekcji  przez ki
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
    for i in range(1, int(filter_order / 2) + 1):
        bi, ai = sig.sos2tf(qsos[:i, :])
        _, hi = sig.freqz(bi, ai)
        ki = np.max(np.abs(hi))
        print('k{} = {}'.format(i, ki))

    _, p, _ = sig.sos2zpk(qsos)
    print("Stabilność - jeżeli moduł kadżdego biegunu jest mniejszy od zera to filtr jest stabilny: \n", np.abs(p))

    # jedna funkcja
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
    return filter_order, sos_, qsos


def main():
    signal_frequency, amplitude = get_signal_params()
    fs, ts = process_sample_rate(signal_frequency)
    filter_order, sos, qsos = filter_designing(fs)

    # druga funkcja
    np.set_printoptions(threshold=np.nan)
    t = np.arange(0, 1, ts)  # time vector
    signal = amplitude * np.sin(2 * np.pi * signal_frequency * t)
    # with open("sin.txt", "w") as data_file:
    #     for sample in signal:
    #         data_file.write(str(sample)+'\n')
    delay = np.array([0.0, 0.0], dtype=float)
    for i in range(0, int(filter_order/2)):
        signal, delay = DSPF_sp_biquad(signal, sos[i], delay)
    display_signal(signal, t, fs)
    print("Parameters: \n", sos)
    plt.show()


if __name__ == "__main__":
    main()
