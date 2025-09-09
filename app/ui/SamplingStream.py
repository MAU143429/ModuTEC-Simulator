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
        if not getattr(self.appstate, "timer_active", False):
            self.appstate.timer_active = True
            if hasattr(self, "after") and hasattr(self, "on_timer"):
                self.after(16, self.on_timer)

    def pause_stream(self):
        self.appstate.paused = True

    def reader_loop(self):
        """Lee WAV por bloques y lo manda a la UI."""
        try:
            with wave.open(self.appstate.audio_file_path, 'rb') as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()

                if sr != self.appstate.sample_rate:
                    self.appstate.sample_rate = sr
                    if hasattr(self, "reset_ring"):
                        self.reset_ring()

                try:
                    wf.setpos(max(0, int(self.appstate.pos_frames)))
                except Exception:
                    self.appstate.pos_frames = 0

                frames_per_chunk = int(self.appstate.block_size)

                while not self.appstate.stop_reader:
                    if self.appstate.paused:
                        time.sleep(0.05)
                        continue

                    data = wf.readframes(frames_per_chunk)
                    if not data:
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

                    try:
                        self.appstate.q.put_nowait(chunk)
                    except queue.Full:
                        try: _ = self.appstate.q.get_nowait()
                        except queue.Empty: pass
                        try: self.appstate.q.put_nowait(chunk)
                        except queue.Full: pass

                    time.sleep(len(chunk) / float(self.appstate.sample_rate))
                    self.appstate.pos_frames += len(chunk)
        finally:
            pass

    def reset_ring(self):
        """Reinicia el ring buffer y reajusta ejes/linea según SR y ventana."""
        n = max(1024, int(self.appstate.window_seconds * float(self.appstate.sample_rate)))
        self.appstate.ring = np.zeros(n, dtype=np.float32)
        line1 = getattr(self, "line1", None)
        ax1 = getattr(self, "ax1", None)
        canvas1 = getattr(self, "canvas1", None)
        if line1:
            x = np.arange(len(self.appstate.ring))
            line1.set_data(x, self.appstate.ring)
            if ax1:
                ax1.set_xlim(0, len(self.appstate.ring))
                ax1.set_ylim(-1.1, 1.1)
            if canvas1:
                canvas1.draw_idle() 

    def on_timer(self):
        """Actualiza ring y gráfica; reprograma el timer."""
        drained = False
        while True:
            try:
                chunk = self.appstate.q.get_nowait()
            except queue.Empty:
                break
            drained = True
            ring = getattr(self.appstate, "ring", None)
            if ring is not None:
                if len(chunk) >= len(ring):
                    ring[:] = chunk[-len(ring):]
                else:
                    L = len(chunk)
                    ring = np.roll(ring, -L)
                    ring[-L:] = chunk
                    self.appstate.ring = ring

        line1 = getattr(self, "line1", None)
        canvas1 = getattr(self, "canvas1", None)
        if drained and line1 and hasattr(self.appstate, "ring"):
            line1.set_ydata(self.appstate.ring)
            if canvas1:
                canvas1.draw_idle()

        if getattr(self.appstate, "reader_thread", None) is not None:
            self.after(16, self.on_timer)

    def after(self, ms, callback):
        if hasattr(self.appstate, "after"):
            self.after(ms, callback)
