import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from scipy import log10, unwrap, arctan2, imag, real


def filter_response(b, a, fs, signal_length):
    frq = np.arange(signal_length) / np.true_divide(signal_length, fs)
    w, h = sig.freqz(b, a, signal_length)
    h_dB = 20 * log10(abs(h))
    plt.figure(3)
    plt.subplot(211)
    plt.plot(frq/2, h_dB)
    plt.ylabel('Wzmocnienie [dB]')
    plt.xlabel("Częstotliwość [Hz]")
    plt.title("Charakterystyka częstotliwościowa ")
    plt.grid()
    plt.subplot(212)
    h_Phase = unwrap(arctan2(imag(h), real(h)))
    plt.plot(frq/2, np.degrees(h_Phase))
    plt.ylabel("Faza [stopnie]")
    plt.xlabel("Częstotliwość [Hz]")
    plt.title(r"Charakterystyka fazowa")
    plt.grid()
    plt.subplots_adjust(hspace=0.5)


def DSPF_sp_biquad(x, b, a, delay):

    a1 = a[1]
    a2 = a[2]
    b0 = b[0]
    b1 = b[1]
    b2 = b[2]
    d0 = delay[0]
    d1 = delay[1]
    r = np.empty([len(x)], dtype=float)
    for i in range(0, len(x)-2):
        x_i = x[i]
        r[i] = b0 * x_i + d0
        d0 = b1 * x_i - a1 * r[i] + d1
        d1 = b2 * x_i - a2 * r[i]

    delay[0] = d0
    delay[1] = d1
    return r, delay


def display_signal(signal, fs, time_vector, figure_number, extra_title='.'):
    signal_length = len(signal)
    frq = np.arange(signal_length) / np.true_divide(signal_length, fs)
    frq = frq[:int(signal_length / 2)]
    y = np.fft.fft(signal) / signal_length
    y = y[:int(signal_length / 2)]
    plt.figure(figure_number)
    plt.plot(time_vector[:signal_length], signal, drawstyle='steps-post')
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.title('Przebieg sygnału nadawanego' + extra_title)
    plt.grid()
    plt.figure(figure_number+1)
    _, _, baseline = plt.stem(frq, abs(y), '-.')
    plt.setp(baseline, 'color', 'r', 'linewidth', 2)
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('|Y(częstotliwość )|')
    plt.title('Widmo amplitudowe nadawanego sygnału' + extra_title)
    plt.grid()


def main():
    Fs = 1000  # sampling rate
    Ts = 1 / Fs  # sampling interval
    t = np.arange(0, 2, 2*Ts)  # time vector
    # 8,14,20
    N = 8
    ff = 50  # frequency of the signal
    signal = np.sin(2*np.pi*ff*t)
    signal_length = len(signal)  # length of the signal
    display_signal(signal, Fs, t, 1)
    char = 'lowpass'
    type = 'butter'
    b, a = sig.iirfilter(N, 100/Fs, btype=char, ftype=type)
    b = [0.25242514, 0.50485028, 0.25242514, 0.2730048, 0.54600961, 0.2730048, 0.32142554, 0.64285108, 0.32142554,
         0.41837842, 0.83675684, 0.41837842]
    a = [1.0, -0, 0.00970056, 1.0, -0, 0.09201921, 1.0, -0, 0.28570215, 1.0, -0,
         0.67351368]

    filter_response(b, a, Fs, signal_length)
    filtered_signal = sig.filtfilt(b, a, signal)
    display_signal(filtered_signal, Fs, t, 4)

    print(b)
    # print(len(b))
    # print(len(b)/3)
    print(a)
    test = signal


    delay = np.array([0.0, 0.0], dtype=float)
    for i in range(0, 4):
        print(b[i*3:(i+1)*3])
        test, delay = DSPF_sp_biquad(test, b[i*3:(i+1)*3], a[i*3:(i+1)*3], delay)
        delay=delay
    # display_signal(test, Fs, t, 6)
    plt.figure(6)
    plt.plot(t,test, drawstyle='steps-post')
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()