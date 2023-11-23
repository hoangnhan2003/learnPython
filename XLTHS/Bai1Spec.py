import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.signal.windows import gaussian
from matplotlib.colors import LogNorm

folders = ['32', '33', '34', '35']
vowels = ['a', 'e', 'i', 'o', 'u']

fslist = {}
xlist = {}


def plot_gaussian_spectrogram(x, fs, window_dur=0.005, dyn_range=120, ax=None):
    step_dur = window_dur / np.sqrt(np.pi) / 8.
    window_nsamp = int(window_dur * fs * 2)
    step_nsamp = int(step_dur * fs)
    window_sigma = (window_nsamp + 1) / 6
    window = gaussian(window_nsamp, window_sigma)
    noverlap = window_nsamp - step_nsamp
    freqs, times, power = spectrogram(x, detrend=False, mode='psd', fs=fs,
                                      scaling='density', noverlap=noverlap,
                                      window=window, nperseg=window_nsamp)
    p_ref = 2e-5
    dB_max = 10 * np.log10(power.max() / (p_ref ** 2))
    vmin = p_ref * 10 ** ((dB_max - dyn_range) / 10)
    cmap = plt.colormaps['Greys']
    if ax is None:
        fig, ax = plt.subplots()
    extent = (times.min(), times.max(), freqs.min(), freqs.max())

    ax.imshow(power, origin='lower', aspect='auto', cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=None), extent=extent)
    return ax

if __name__ == "__main__":
    for folder in folders:
        for vowel in vowels:
            path = folder + '/' + vowel + '.wav'
            fs, x = wavfile.read(path)
            fslist[path] = fs
            xlist[path] = x

    # Hien thi pho am A
    fig, axs = plt.subplots(2, 2, constrained_layout=True);
    axs = axs.flatten()

    fig.suptitle("Vowel A", fontsize=16);
    i = 0
    for folder in folders:
      path = folder + '/a.wav'
      ax = plot_gaussian_spectrogram(xlist[path], fslist[path], ax=axs[i])
      ax.set_ylim(0, 12000)
      ax.set_xlabel('time (s)')
      ax.set_ylabel('frequency (Hz)')
      ax.set_title('Spectrogram of ' + path)
      i = i + 1
    plt.show()
    # Hien thi pho am E
    fig, axs = plt.subplots(2, 2, constrained_layout=True);
    axs = axs.flatten()

    fig.suptitle("Vowel E", fontsize=16);
    i = 0;
    for folder in folders:
      path = folder + '/e.wav';
      ax = plot_gaussian_spectrogram(xlist[path], fslist[path], ax=axs[i])
      ax.set_ylim(0, 12000)
      ax.set_xlabel('time (s)')
      ax.set_ylabel('frequency (Hz)')
      ax.set_title('Spectrogram of ' + path);
      i = i + 1;
    plt.show()

    # Lam tuong tự cho các âm còn lại