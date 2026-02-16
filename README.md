# Pulse-Code-Modulation
# Aim
Write a simple Python program for the modulation and demodulation of PCM, and DM.
# Tools required
Python IDE with NumPy and SciPy


# Program
# Pulse Code Modulation
```
import numpy as np
import matplotlib.pyplot as plt

sampling_rate = 5000
frequency = 50
duration = 0.1
quantization_levels = 16

t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

message_signal = np.sin(2 * np.pi * frequency * t)

clock_signal = np.sign(np.sin(2 * np.pi * 200 * t))

quantization_step = (max(message_signal) - min(message_signal)) / quantization_levels
quantized_signal = np.round(message_signal / quantization_step) * quantization_step

pcm_signal = (quantized_signal - min(quantized_signal)) / quantization_step
pcm_signal = pcm_signal.astype(int)

plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label="Message Signal (Analog)", color='blue')
plt.title("Message Signal (Analog)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, clock_signal, label="Clock Signal (Increased Frequency)", color='green')
plt.title("Clock Signal (Increased Frequency)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.step(t, quantized_signal, label="PCM Modulated Signal", color='red')
plt.title("PCM Modulated Signal (Quantized)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, quantized_signal, label="PCM Demodulation Signal", color='purple', linestyle='--')
plt.title("PCM Demodulation Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
```
# Delta-Modulation
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

fs = 10000
f = 10
T = 1
delta = 0.1

t = np.arange(0, T, 1/fs)
message_signal = np.sin(2 * np.pi * f * t)

encoded_signal = np.zeros(len(t))
dm_output = np.zeros(len(t))

for i in range(1, len(t)):
    if message_signal[i] > dm_output[i-1]:
        encoded_signal[i] = 1
        dm_output[i] = dm_output[i-1] + delta
    else:
        encoded_signal[i] = 0
        dm_output[i] = dm_output[i-1] - delta

def low_pass_filter(signal, cutoff_freq, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)

filtered_signal = low_pass_filter(dm_output, cutoff_freq=12, fs=fs)

plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t, message_signal)
plt.title("Original Signal")
plt.grid()

plt.subplot(3, 1, 2)
plt.step(t, dm_output, where='mid')
plt.title("Delta Modulated Signal")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, filtered_signal, linestyle='dotted')
plt.title("Demodulated & Filtered Signal")
plt.grid()

plt.tight_layout()
plt.show()
```
# Output Waveform
#PULSE CODE MODULATION
<img width="1189" height="990" alt="image" src="https://github.com/user-attachments/assets/abb7f701-4c7f-4bd2-8b81-42b47ee2be25" />



#DELTA MODULATION
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/3ec28c05-8408-49b4-9602-b90aaf8b6d25" />



# Results
The analog signal was successfully encoded and reconstructed using PCM and DM techniques in Python, verifying their working principles..
