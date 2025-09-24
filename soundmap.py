from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal, pyqtSlot
import numpy as np
import soundfile as sf
import pyqtgraph as pg
from collections import deque
import audioread  # fallback decoder for MP3


class FileLoader(QtCore.QThread):
    chunkReady = pyqtSignal(np.ndarray, int)  # (audio_chunk mono float32, samplerate)
    metaReady = pyqtSignal(int, int)          # (samplerate, channels)
    error = pyqtSignal(str)
    finishedOK = pyqtSignal()

    def __init__(self, path: str, block_frames: int = 4096, parent=None):
        super().__init__(parent)
        self.path = path
        self.block_frames = block_frames
        self._stop = False

    def stop(self):
        self._stop = True

    def _stream_with_soundfile(self):
        with sf.SoundFile(self.path, mode='r') as f:
            sr = f.samplerate
            ch = f.channels
            self.metaReady.emit(sr, ch)
            while not self._stop:
                data = f.read(self.block_frames, dtype='float32', always_2d=True)
                if data.size == 0:
                    break
                mono = data.mean(axis=1)
                self.chunkReady.emit(mono, sr)

    def _stream_with_audioread(self):
        with audioread.audio_open(self.path) as f:
            sr = f.samplerate
            ch = f.channels
            self.metaReady.emit(sr, ch)
            frame_bytes_per_sample = 2  # audioread yields 16-bit PCM
            block_samples = self.block_frames * ch
            cache = bytearray()
            for buf in f:
                if self._stop:
                    break
                cache.extend(buf)
                bytes_per_frame = frame_bytes_per_sample * ch
                while not self._stop and len(cache) >= block_samples * frame_bytes_per_sample:
                    take = block_samples * frame_bytes_per_sample
                    chunk = cache[:take]
                    del cache[:take]
                    arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    if ch > 1:
                        arr = arr.reshape(-1, ch).mean(axis=1)
                    self.chunkReady.emit(arr, sr)
            if not self._stop and len(cache) > 0:
                arr = np.frombuffer(cache, dtype=np.int16).astype(np.float32) / 32768.0
                if ch > 1:
                    pad = (-arr.size) % ch
                    if pad:
                        arr = np.pad(arr, (0, pad))
                    arr = arr.reshape(-1, ch).mean(axis=1)
                self.chunkReady.emit(arr, sr)

    def run(self):
        try:
            try:
                self._stream_with_soundfile()
            except Exception:
                self._stream_with_audioread()
            self.finishedOK.emit()
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Loader (No Freeze) — Demo")
        self.resize(900, 500)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        top_bar = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load audio…")
        self.status_lbl = QtWidgets.QLabel("Idle")
        self.status_lbl.setStyleSheet("color: gray")
        top_bar.addWidget(self.btn_load)
        top_bar.addStretch(1)
        top_bar.addWidget(self.status_lbl)

        self.plot = pg.PlotWidget()
        self.plot.setYRange(-1.1, 1.1)
        self.plot.enableAutoRange(x=True, y=False)
        self.plot.showGrid(x=True, y=True)
        self.curve = self.plot.plot(np.zeros(1000, dtype=np.float32))

        layout.addLayout(top_bar)
        layout.addWidget(self.plot, 1)

        self.sample_rate = 48000
        self.buffer_seconds = 2.0
        self.buffer = deque(maxlen=int(self.sample_rate * self.buffer_seconds))

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start()

        self.btn_load.clicked.connect(self._load_clicked)
        self.loader = None

    @pyqtSlot()
    def _load_clicked(self):
        dlg = QtWidgets.QFileDialog(self, "Select audio file")
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dlg.setNameFilters([
            "Audio files (*.wav *.flac *.ogg *.aiff *.aif *.mp3)",
            "All files (*.*)",
        ])
        if dlg.exec():
            path = dlg.selectedFiles()[0]
            self._start_loader(path)

    def _start_loader(self, path: str):
        self.buffer.clear()
        self.status_lbl.setText(f"Loading: {QtCore.QFileInfo(path).fileName()}")
        self.status_lbl.setStyleSheet("color: orange")

        if self.loader is not None and self.loader.isRunning():
            self.loader.stop()
            self.loader.wait(1000)

        self.loader = FileLoader(path)
        self.loader.chunkReady.connect(self._on_chunk)
        self.loader.metaReady.connect(self._on_meta)
        self.loader.error.connect(self._on_error)
        self.loader.finishedOK.connect(self._on_finished)
        self.loader.start()

    @pyqtSlot(int, int)
    def _on_meta(self, sr: int, ch: int):
        self.sample_rate = sr
        self.buffer = deque(maxlen=int(self.sample_rate * self.buffer_seconds))
        self.status_lbl.setText(f"Streaming… {sr} Hz / {ch} ch")
        self.status_lbl.setStyleSheet("color: dodgerblue")

    @pyqtSlot(np.ndarray, int)
    def _on_chunk(self, mono: np.ndarray, sr: int):
        self.buffer.extend(mono.tolist())

    @pyqtSlot()
    def _on_finished(self):
        self.status_lbl.setText("Loaded (end of file)")
        self.status_lbl.setStyleSheet("color: green")

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self.status_lbl.setText(f"Error: {msg}")
        self.status_lbl.setStyleSheet("color: crimson")

    @pyqtSlot()
    def _on_timer(self):
        if len(self.buffer) > 0:
            data = np.fromiter(self.buffer, dtype=np.float32, count=len(self.buffer))
            if data.size > 8000:
                step = max(1, data.size // 4000)
                data = data[::step]
            self.curve.setData(data)
            self.plot.enableAutoRange(x=True, y=False)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()
