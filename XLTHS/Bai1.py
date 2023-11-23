import numpy as np
import matplotlib.pyplot as plt
import statistics as stts
from scipy.io import wavfile
from scipy.signal.windows import gaussian
from scipy.signal import get_window
from matplotlib.colors import LogNorm
from scipy.signal import spectrogram, find_peaks
import librosa
from scipy.signal import lfilter, hamming, resample
import math

Fs, x = wavfile.read('a.wav')
duration = len(x) / Fs

window_size = round(0.01 * Fs)  # Size of the analysis window
print(window_size)

# MA
xMA = np.zeros_like(x, dtype=float)
for i in range(len(x)):
    for m in range(window_size):
        if (i - m >= 0):
            xMA[i] += np.abs(x[i - m])

xMA /= np.max(xMA)

# Plot signal and MA
plt.figure(figsize=(12, 6))

plt.plot(np.arange(len(x)) / Fs, x / np.max(x), 'b', linewidth=2)
plt.plot(np.arange(len(x)) / Fs, xMA, 'r--', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Signal & MA')
plt.legend(['x[n]', 'MA'])
plt.show()

x = x[xMA > 0.1]

plt.plot(np.arange(len(x)) / Fs, x / np.max(x), 'b', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Signal & MA')
plt.legend(['x[n]', 'MA'])
plt.show()

step_dur = None;
window_dur = 0.005
dyn_range = 120;
cmap = None;
ax = None;


# set default for step_dur, if unspecified. This value is optimal for Gaussian windows.
if step_dur is None:
    #Nếu biến step_dur không được xác định trước, đoạn code sẽ đặt giá trị mặc định cho nó bằng cách
    # chia window_dur cho căn bậc hai của pi chia cho 8. Giá trị này là tối ưu cho cửa sổ Gaussian,
    # là một hàm có hình dạng chuông, có độ rộng phụ thuộc vào tham số sigma. Cửa sổ Gaussian sẽ được sử
    # dụng để áp dụng cho các khung phân tích âm thanh.
    step_dur = window_dur / np.sqrt(np.pi) / 8.

# convert window & step durations from seconds to numbers of samples (which is what
# scipy.signal.spectrogram takes as input).
window_nsamp = int(window_dur * Fs * 2)
step_nsamp = int(step_dur * Fs)

# make the window. A Gaussian filter needs a minimum of 6σ - 1 samples, so working
# backward from window_nsamp we can calculate σ.
window_sigma = (window_nsamp + 1) / 6
window = get_window(('gaussian', window_sigma), window_nsamp)

# convert step size into number of overlapping samples in adjacent analysis frames
noverlap = window_nsamp - step_nsamp

# compute the power spectral density
freqs, times, power = spectrogram(x, detrend=False, mode='psd', fs=Fs,
                                  scaling='density', noverlap=noverlap,
                                  window=window, nperseg=window_nsamp)

p_ref = 2e-5  # 20 μPa, the standard reference pressure for sound in air

# set lower bound of colormap (vmin) from dynamic range. The upper bound defaults
# to the largest value in the spectrogram, so we don't need to set it explicitly.
dB_max = 10 * np.log10(power.max() / (p_ref ** 2))
vmin = p_ref * 10 ** ((dB_max - dyn_range) / 10)

# set default colormap, if none specified
if cmap is None:
    cmap = plt.colormaps['Greys']
# or if cmap is a string, get the actual object
elif isinstance(cmap, str):
    cmap = plt.colormaps[cmap]

def get_formants(x, Fs):

  x= resample(x,8000);

  Fs=8000;
  # Get Hamming window
  N = len(x)
  w = np.hamming(N)

  # Apply window and high pass filter.
  x1 = x * w
  x1 = lfilter([1], [1, 0.63], x1)

  # Get LPC.
  ncoeff = int(2 + Fs / 1000);
  A = librosa.lpc(y = x1, order = ncoeff)

  # Get roots.
  rts = np.roots(A)
  rts = [r for r in rts if np.imag(r) >= 0]

  # Get angles.
  angz = np.arctan2(np.imag(rts), np.real(rts))

  # Get frequencies.
  frqs = sorted(angz * (Fs / (2 * math.pi)))
  indices = np.argsort(angz * (Fs / (2 * math.pi))).tolist();
  bw = [-1/2 * (Fs / (2 * np.pi)) * np.log(np.abs(rts[i])) for i in indices]

  nn = 1
  formants = []

  for kk in range(len(frqs)):
    if 90 < frqs[kk] and bw[kk] < 400:
      formants.append(frqs[kk])
      nn += 1
      # if 90 < frqs[kk] and bw[kk] < 400:
      #     formants.append(frqs[kk])
      #     nn += 1


  return formants;

formants = [[], [], [], [], [], [], [], [], [],
            [], [], [], [], [], [], [], [], []];

transposedPower = power.T;
for k in range(len(transposedPower)): # foreach FFT from spectrogram
  fmts = get_formants(transposedPower[k], Fs);
  if k != 0:
    for j in range(len(formants)):
      formants[j].append(formants[j][k - 1]);
  else:
    for j in range(len(formants)):
      formants[j].append(0);

  for j in range(len(fmts)): # foreach
    formants[j][k] = fmts[j];

for k in range(len(formants)):
  print(np.mean(formants[k]))
  print(np.std(formants[k]))

# other arguments to the figure
extent = (times.min(), times.max(), freqs.min(), freqs.max())

# plot
plt.imshow(power, origin='lower', aspect='auto', cmap=cmap,
          norm=LogNorm(vmin=vmin, vmax=None), extent=extent)

# formants
plt.plot(times, formants[0], color='r', linewidth=1);
plt.plot(times, formants[1], color='g', linewidth=1);
plt.plot(times, formants[2], color='b', linewidth=1);
plt.plot(times, formants[3], color='#eb8', linewidth=1);

# optimized graph
plt.ylim(0, 12000);
plt.title("Spectrogram of Signal");
plt.xlabel('time (s)');
plt.ylabel('frequency (Hz)');
plt.legend();

plt.show();