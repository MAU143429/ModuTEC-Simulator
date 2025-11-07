
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

    def recommend_params(self):
        
        # General Information
        try:
            src_path = self.state.audio_file_path
            if not src_path or not src_path.lower().endswith(".wav"):
                raise ValueError("se esperaba un archivo .wav")
            with wave.open(src_path, "rb") as wf:
                Fs = wf.getframerate()
        except Exception as e:
            print(f"[audio] no se pudo extraer sample rate del WAV: {e}")         
        fmax = self.estimate_fmax_from_wav()
        self.state.fmax = fmax
        self.state.recommended_Fs = Fs
        
        # AM Values
        
        self.state.recommended_am_fc = min(Fs/4, max(100, 10*fmax))
        
        self.state.recommended_am_mu = 0.7

        self.state.recommended_am_Ac = 0.90 / (1 + self.state.recommended_am_mu)

        # FM Values

        self.state.recommended_fm_fc = min(Fs/4, max(100, 10*fmax))
        
        self.state.recommended_fm_Ac = 0.90
        
        beta = 2.0  # índice base propuesto
        BW = 2 * (beta * fmax + fmax)  # ancho de banda según Carson
        nyquist_margin = 0.9 * (Fs / 2)

        if BW > nyquist_margin:
            beta_max = (0.45 * Fs - fmax) / fmax   # valor máximo permitido
            self.state.recommended_fm_beta = max(1.0, min(beta, beta_max))   # mantenerlo entre 1 y beta_max
        
        # ASK Values
        
        self.state.recommended_ask_bitrate = min(Fs/10, 4*fmax)
        
        self.state.recommended_ask_fc = min(Fs/4, max(1000, 10*self.state.recommended_ask_bitrate))

        self.state.recommended_ask_Ac = 0.90
        
        # FSK Values

        self.state.recommended_fsk_bitrate = min(Fs/10, 4*fmax)

        self.state.recommended_fsk_fc1 = min(Fs/4, max(1000, 10*self.state.recommended_fsk_bitrate))
        
        self.state.recommended_fsk_fc2 = self.state.recommended_fsk_fc1 * 5

        self.state.recommended_fsk_Ac = 0.90

    def estimate_fmax_from_wav(self, chunks=6, chunk_frames=44100, target_fs=8000, nfft=4096, energy_pct=0.99):
        """
        Lightweight Fmax estimator:
        - samples `chunks` blocks evenly across the file
        - decimates to `target_fs` (integer decimation)
        - computes averaged periodograms (Hann, 50% overlap)
        - returns frequency where cumulative energy >= energy_pct
        """
        try:
            wf = wave.open(self.state.audio_file_path, "rb")
        except Exception:
            return 0.0
        try:
            Fs = wf.getframerate()
            nch = wf.getnchannels()
            sampw = wf.getsampwidth()
            dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampw, np.int16)
            total_frames = wf.getnframes()
            if total_frames <= 0:
                return 0.0

            psd_acc = None
            wins = 0
            # sample `chunks` positions across the file
            chunks = max(1, int(chunks))
            for k in range(chunks):
                if chunks == 1:
                    pos = 0
                else:
                    pos = int((k * max(0, total_frames - chunk_frames)) / max(1, chunks - 1))
                try:
                    wf.setpos(min(pos, total_frames - 1))
                except Exception:
                    wf.rewind()
                raw = wf.readframes(chunk_frames)
                if not raw:
                    continue
                data = np.frombuffer(raw, dtype=dtype)
                if nch > 1:
                    data = data.reshape(-1, nch).mean(axis=1)
                x = data.astype(np.float32)
                # normalize integer types
                if np.issubdtype(dtype, np.integer):
                    x /= float(np.iinfo(dtype).max)

                # integer decimation to speed up (no anti-aliasing filter for simplicity)
                decim = max(1, int(Fs // target_fs)) if (target_fs and target_fs < Fs) else 1
                if decim > 1:
                    x = x[::decim]
                    Fs_eff = Fs / decim
                else:
                    Fs_eff = Fs

                step = nfft // 2
                if len(x) < nfft:
                    # pad to nfft
                    seg = np.zeros(nfft, dtype=np.float32)
                    seg[:len(x)] = x
                    seg *= np.hanning(nfft)
                    X = np.fft.rfft(seg)
                    P = (np.abs(X) ** 2) / nfft
                    psd_acc = P if psd_acc is None else psd_acc + P
                    wins += 1
                else:
                    for i in range(0, len(x) - nfft + 1, step):
                        seg = x[i:i + nfft] * np.hanning(nfft)
                        X = np.fft.rfft(seg)
                        P = (np.abs(X) ** 2) / nfft
                        psd_acc = P if psd_acc is None else psd_acc + P
                        wins += 1

            if wins == 0 or psd_acc is None:
                return 0.0

            psd = psd_acc / wins
            freqs = np.fft.rfftfreq(nfft, d=1.0 / Fs_eff)
            psd = np.maximum(psd, 0.0)
            cum = np.cumsum(psd)
            total = cum[-1] if cum.size and cum[-1] > 0 else 1.0
            idx = np.searchsorted(cum, energy_pct * total)
            fmax = float(freqs[min(idx, len(freqs) - 1)])
            return fmax
        finally:
            try:
                wf.close()
            except Exception:
                pass
    
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