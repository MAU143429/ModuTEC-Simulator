import numpy as np
import wave

# Parameters
duration = 20.0        # seconds
freq = 20.0          # Hz
sample_rate = 44100   # samples per second
amplitude = 32767     # max for int16

# Generate time array
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
# Generate sine wave
signal = amplitude * np.sin(2 * np.pi * freq * t)
signal = signal.astype(np.int16)

# Write to .wav file
with wave.open('assets/10hztone.wav', 'wb') as wav_file:
    wav_file.setnchannels(1)        # mono
    wav_file.setsampwidth(2)        # 2 bytes per sample (16 bits)
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(signal.tobytes())