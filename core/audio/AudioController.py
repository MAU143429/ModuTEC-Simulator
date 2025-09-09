
from tkinter import filedialog
import os
import time
import threading
import queue
import wave
import numpy as np
import sounddevice as sd
from pydub import AudioSegment


class AudioController:
    
    def __init__(self, appstate):
        self.state = appstate
        self._sd_stream = None
        self._q = queue.Queue(maxsize=10)


    def mp3ToWav(self, mp3_path):
        """Convierte un archivo MP3 a WAV temporal y retorna la ruta WAV."""
        mp3_audio = AudioSegment.from_mp3(mp3_path)
        wav_path = mp3_path + ".temp.wav"
        mp3_audio.export(wav_path, format="wav")
        return wav_path


    
    def ensure_sd_stream(self):
        """Abre/inicia el stream de salida si no existe."""
        if self.sd_stream is not None:
            return
        try:
            self.sd_stream = sd.OutputStream(
                samplerate=int(self.state.sample_rate),
                channels=self.state.audio_channels,
                dtype="float32",
                blocksize=int(self.state.blocksize),
                device=self.state.sd_device,
            )
            self.sd_stream.start()
            print("[audio] output stream started")
        except Exception as e:
            print(f"[audio] no se pudo abrir el output stream: {e}")
            self.sd_stream = None