import numpy as np
import threading
import queue
import time
import wave

class SamplingStream:
    def __init__(self, appstate):
        self.appstate = appstate


    def start_stream(self):
        if not self.appstate.audio_file_path:
            return
        if self.appstate.reader_thread is None or not self.appstate.reader_thread.is_alive():
            self.appstate.stop_reader = False
            self.appstate.reader_thread = threading.Thread(target=self.reader_loop, daemon=True)
            self.appstate.reader_thread.start()
        self.appstate.paused = False


    def pause_stream(self):
        self.appstate.paused = True

    def reader_loop(self):
        try:
            with wave.open(self.appstate.audio_file_path, 'rb') as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()

                if sr != self.appstate.sample_rate:
                    self.appstate.sample_rate = sr
                    self.appstate.needs_reset = True  # <- UI reajusta ring a 5 s

                try:
                    wf.setpos(max(0, int(self.appstate.pos_frames)))
                except Exception:
                    self.appstate.pos_frames = 0

                frames_per_chunk = int(self.appstate.block_size)  # fijo (2048)

                while not self.appstate.stop_reader:
                    if self.appstate.paused:
                        time.sleep(0.05)
                        continue

                    data = wf.readframes(frames_per_chunk)
                    if not data:
                        # fin de archivo: pausar y dejar última vista
                        self.appstate.paused = True
                        break

                    # Convertir a float32 mono (canal 0)
                    if sw == 2:
                        chunk = np.frombuffer(data, dtype=np.int16)
                        if ch > 1:
                            chunk = chunk.reshape(-1, ch)[:, 0]
                        chunk = chunk.astype(np.float32) / 32768.0
                    elif sw == 1:
                        chunk = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
                        if ch > 1:
                            chunk = chunk.reshape(-1, ch)[:, 0]
                        chunk = (chunk - 128.0) / 128.0
                    elif sw == 4:
                        chunk = np.frombuffer(data, dtype=np.float32)
                        if ch > 1:
                            chunk = chunk.reshape(-1, ch)[:, 0]
                    else:
                        print("[audio] formato WAV no soportado (samplewidth)", sw)
                        self.appstate.paused = True
                        break

                    # cola con política "descarta lo más viejo"
                    try:
                        self.appstate.q.put_nowait(chunk)
                    except queue.Full:
                        try: _ = self.appstate.q.get_nowait()
                        except queue.Empty: pass
                        try: self.appstate.q.put_nowait(chunk)
                        except queue.Full: pass

                    # ritmo tiempo real
                    time.sleep(len(chunk) / float(self.appstate.sample_rate))
                    self.appstate.pos_frames += len(chunk)
        finally:
            pass

