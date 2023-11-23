import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as sig

if __name__ == "__main__":
    # fs, x = wav.read("a.wav")
    # lpc_order = 16  # Số thứ tự của bộ lọc lpc
    # a = sig.lpc(x, lpc_order) # Hệ số lpc
    # r, p, k = sig.residuez(1, a)  # Cực và số dư
    # p = p[np.imag(p) >= 0]
    # angz = np.arctan2(np.imag(p), np.real(p))  # Góc phức
    # frqs = angz * (fs / (2 * np.pi)) # Tần số
    # bw = -0.5 * (fs / (2 * np.pi)) * np.log(np.abs(p)) # Băng thông
    # plt.figure(figsize=(8, 6))
    # plt.scatter(frqs, bw)
    # plt.xlabel("Tần số(Hz)")
    # plt.ylabel("Băng thông(Hz)")
    # plt.title("Biểu đồ formant")
    # plt.grid()
    # plt.show()
    fs, x = wav.read("a.wav")
    lpc_order = 16  # Số thứ tự của bộ lọc lpc
    b, a = sig.butter(lpc_order, 0.05)  # Hệ số bộ lọc butterworth
    w, h = sig.freqz(b, a)  # Tần số và phản ứng tần số của bộ lọc
    frqs = w * (fs / (2 * np.pi))  # Tần số
    bw = -20 * np.log10(np.abs(h))  # Băng thông
    plt.figure(figsize=(8, 6))
    plt.scatter(frqs, bw)
    plt.xlabel("Tần số(Hz)")
    plt.ylabel("Băng thông(Hz)")
    plt.title("Biểu đồ formant")
    plt.grid()
    plt.show()