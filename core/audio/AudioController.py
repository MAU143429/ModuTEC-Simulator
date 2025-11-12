import queue
import wave
import numpy as np
from pydub import AudioSegment


# Convert dBFS (decibels relative to full scale) to a linear amplitude ratio.
def dbfs_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))

# Compute the RMS (root mean square) of a signal array.
def compute_rms(x):
    if x is None or x.size == 0:
        return 0.0
    x64 = x.astype(np.float64, copy=False)
    x_dc = x64 - np.mean(x64)
    return float(np.sqrt(np.mean(x_dc * x_dc)))

# =====================================================================#
#                         AudioController Class                        #
#                                                                      #
#  - Load and analyze audio files (WAV, with MP3->WAV helper).         #
#  - Persist global audio metadata and recommendations into app state. #
#                                                                      #
# =====================================================================#
class AudioController:
    
    # Constructor
    def __init__(self, appstate):
        self.state = appstate
        self._sd_stream = None
        self._q = queue.Queue(maxsize=10)

    # Convert MP3 to WAV using pydub
    def mp3ToWav(self, mp3_path):
        print("path mp3:", mp3_path)
        mp3_audio = AudioSegment.from_mp3(mp3_path)
        wav_path = mp3_path + ".temp.wav"
        mp3_audio.export(wav_path, format="wav")
        return wav_path

    # Load WAV file and analyze its properties
    def load_wav_and_analyze(self, wav_path):
        try:
            wf = wave.open(wav_path, "rb")
            Fs = wf.getframerate()
            nch = wf.getnchannels()
            sampw = wf.getsampwidth()
            dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampw, np.int16)

            raw = wf.readframes(wf.getnframes())
            data = np.frombuffer(raw, dtype=dtype)

            if nch > 1:
                data = data.reshape(-1, nch).mean(axis=1)

            if np.issubdtype(dtype, np.integer):
                data = data.astype(np.float32) / np.iinfo(dtype).max
            else:
                data = data.astype(np.float32, copy=False)

            wf.close()

            self.state.audio_file_path = wav_path
            self.state.audio_array = data
            self.state.sample_rate = Fs
            self.state.num_samples = len(data)
            self.state.audio_channels = nch

            rms_in = compute_rms(data)
            self.state.rms_in = float(rms_in)
            self.state.rms_target = float(dbfs_to_linear(-12.0))
            self.state.global_gain = 1.0   # NO usar en el pipeline adaptativo
            self.state.normalize_mode = "block"

            if not hasattr(self.state, "display_gain"):
                self.state.display_gain = 1.0

            print(f"[audio] loaded: Fs={Fs}Hz, ch={nch}, RMS_in={rms_in:.6f}")
        except Exception as e:
            print(f"[audio] error al cargar WAV: {e}")
    
    # Recommend modulation parameters based on audio file analysis
    def recommend_params(self):
        try:
            src_path = self.state.audio_file_path
            if not src_path or not src_path.lower().endswith(".wav"):
                raise ValueError("se esperaba un archivo .wav")
            with wave.open(src_path, "rb") as wf:
                Fs = wf.getframerate()
        except Exception as e:
            print(f"[audio] no se pudo extraer sample rate del WAV: {e}")
            Fs = self.state.sample_rate or 44100

        fmax = self.estimate_fmax_from_wav()
        self.state.fmax = fmax
        self.state.recommended_Fs = Fs

        # AM
        self.state.recommended_am_fc = min(Fs/4, max(100, 10*fmax))
        self.state.recommended_am_mu = 0.7
        self.state.recommended_am_Ac = 0.90 / (1 + self.state.recommended_am_mu)

        # FM
        self.state.recommended_fm_fc = min(Fs/4, max(100, 10*fmax))
        self.state.recommended_fm_Ac = 0.90
        self.state.recommended_fm_beta = 1.0

        # ASK
        self.state.recommended_ask_bitrate = np.clip(Fs / 24.0, 800.0, 5000.0)
        self.state.recommended_ask_fc = np.clip(8.0 * self.state.recommended_ask_bitrate, 2000.0, Fs / 5.0)
        self.state.recommended_ask_Ac = 0.9
    
        # FSK
        self.state.recommended_fsk_bitrate = np.clip(Fs / 24.0, 800.0, 5000.0)
        self.state.recommended_fsk_fc2 = np.clip(6.0 * self.state.recommended_fsk_bitrate , 2000.0, Fs / 6.0)
        self.state.recommended_fsk_fc1 = self.state.recommended_fsk_fc2 + max(0.5 * self.state.recommended_fsk_bitrate, 1500.0)
        self.state.recommended_fsk_Ac = 0.9

    # Estimate maximum frequency component from WAV file
    def estimate_fmax_from_wav(self, chunks=6, chunk_frames=44100, target_fs=8000, nfft=4096, energy_pct=0.99):
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
            chunks = max(1, int(chunks))
            for k in range(chunks):
                pos = int((k * max(0, total_frames - chunk_frames)) / max(1, chunks - 1)) if chunks > 1 else 0
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
                if np.issubdtype(dtype, np.integer):
                    x /= float(np.iinfo(dtype).max)

                decim = max(1, int(Fs // (target_fs or Fs))) if target_fs and target_fs < Fs else 1
                if decim > 1:
                    x = x[::decim]
                    Fs_eff = Fs / decim
                else:
                    Fs_eff = Fs

                step = nfft // 2
                if len(x) < nfft:
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
