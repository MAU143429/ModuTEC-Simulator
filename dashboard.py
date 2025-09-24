#!/usr/bin/env python3
"""
Real‑time Audio Visualizer (PyQt6 + pyqtgraph + sounddevice)
- Waveform and FFT spectrum
- Start/Stop controls and input device selection
- Low-latency by using a small blocksize and a ring buffer

Dependencies:
    pip install PyQt6 pyqtgraph sounddevice numpy
"""
from __future__ import annotations

import sys
import queue
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import sounddevice as sd
import pyqtgraph as pg

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSpinBox, QVBoxLayout, QWidget
)


# ---------- Audio Backend ----------
@dataclass
class AudioConfig:
    samplerate: int = 44100
    blocksize: int = 512          # Lower = lower latency, higher CPU
    channels: int = 1
    device_index: Optional[int] = None  # None = default input device
    seconds_in_buffer: float = 2.0      # Visible history for waveform
    fft_size: int = 8192                # Power of 2 recommended


class AudioStream:
    """Captures audio from microphone using sounddevice and pushes chunks to a queue."""
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
        self._stream: Optional[sd.InputStream] = None

    # PortAudio callback runs in another thread
    def _callback(self, indata, frames, time, status):
        if status:
            # Status could be InputUnderflow or Overflow; we avoid printing repeatedly
            pass
        # Copy to avoid referencing the underlying PortAudio buffer
        mono = indata[:, 0].copy()
        try:
            self._q.put_nowait(mono)
        except queue.Full:
            # Drop the chunk if UI isn't keeping up; prevents blocking
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            # Try again (best-effort)
            try:
                self._q.put_nowait(mono)
            except queue.Full:
                pass

    def start(self):
        if self._stream is not None:
            return
        kwargs = dict(
            samplerate=self.cfg.samplerate,
            blocksize=self.cfg.blocksize,
            channels=self.cfg.channels,
            dtype="float32",
            callback=self._callback,
        )
        if self.cfg.device_index is not None:
            kwargs["device"] = self.cfg.device_index
        self._stream = sd.InputStream(**kwargs)
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        # Clear the queue
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def read_all_chunks(self) -> List[np.ndarray]:
        """Drain the queue and return all available chunks."""
        chunks: List[np.ndarray] = []
        while True:
            try:
                chunks.append(self._q.get_nowait())
            except queue.Empty:
                break
        return chunks


# ---------- GUI ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real‑time Audio Visualizer (PyQt)")
        self.resize(1000, 700)

        # Discover audio devices
        self.devices = self._list_input_devices()
        default_sr = int(sd.query_devices(None, "input")["default_samplerate"] or 44100)

        # Config
        self.cfg = AudioConfig(samplerate=default_sr)
        self.audio = AudioStream(self.cfg)

        # Ring buffer for waveform
        self.ring_len = int(self.cfg.seconds_in_buffer * self.cfg.samplerate)
        self.ring = np.zeros(self.ring_len, dtype=np.float32)

        # Widgets
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Controls
        controls = QHBoxLayout()
        layout.addLayout(controls)

        self.device_combo = QComboBox()
        for idx, name in self.devices:
            self.device_combo.addItem(name, idx)
        self.device_combo.setCurrentIndex(self._current_device_index_in_combo())
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)

        self.sr_spin = QSpinBox()
        self.sr_spin.setRange(8000, 192000)
        self.sr_spin.setSingleStep(1000)
        self.sr_spin.setValue(self.cfg.samplerate)
        self.sr_spin.valueChanged.connect(self.on_samplerate_changed)

        self.block_spin = QSpinBox()
        self.block_spin.setRange(64, 4096)
        self.block_spin.setSingleStep(64)
        self.block_spin.setValue(self.cfg.blocksize)
        self.block_spin.valueChanged.connect(self.on_blocksize_changed)

        self.fft_spin = QSpinBox()
        self.fft_spin.setRange(1024, 65536)
        self.fft_spin.setSingleStep(1024)
        self.fft_spin.setValue(self.cfg.fft_size)
        self.fft_spin.valueChanged.connect(self.on_fft_changed)

        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.start_btn.clicked.connect(self.start_audio)
        self.stop_btn.clicked.connect(self.stop_audio)

        controls.addWidget(QLabel("Input:"))
        controls.addWidget(self.device_combo, stretch=2)
        controls.addWidget(QLabel("Rate:"))
        controls.addWidget(self.sr_spin)
        controls.addWidget(QLabel("Block:"))
        controls.addWidget(self.block_spin)
        controls.addWidget(QLabel("FFT:"))
        controls.addWidget(self.fft_spin)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)

        # Plots
        pg.setConfigOptions(antialias=True)
        self.wave_plot = pg.PlotWidget(title="Waveform (last {:.1f}s)".format(self.cfg.seconds_in_buffer))
        self.wave_plot.setLabel("bottom", "Time", units="samples")
        self.wave_plot.setLabel("left", "Amplitude")
        self.wave_curve = self.wave_plot.plot(self.ring, pen="w")
        layout.addWidget(self.wave_plot, stretch=3)

        self.fft_plot = pg.PlotWidget(title="Spectrum (RFFT)")
        self.fft_plot.setLabel("bottom", "Frequency", units="Hz")
        self.fft_plot.setLabel("left", "Magnitude", units="dBFS")
        self.fft_curve = self.fft_plot.plot(pen="c")
        layout.addWidget(self.fft_plot, stretch=2)

        # Timer to pull audio and update plots
        self.timer = QTimer(self)
        self.timer.setInterval(15)  # ms; UI refresh ~60 FPS
        self.timer.timeout.connect(self.update_plots)

        # Start automatically
        self.start_audio()

    # ---- Helpers ----
    def _list_input_devices(self):
        devs = sd.query_devices()
        items = []
        for i, d in enumerate(devs):
            if d["max_input_channels"] > 0:
                name = f"{i}: {d['name']}"
                items.append((i, name))
        # Fallback to default device if none found
        if not items:
            items = [(None, "Default input")]
        return items

    def _current_device_index_in_combo(self) -> int:
        target = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == target:
                return i
        return 0

    # ---- Slots ----
    def start_audio(self):
        self.audio.start()
        if not self.timer.isActive():
            self.timer.start()

    def stop_audio(self):
        self.timer.stop()
        self.audio.stop()

    def on_device_changed(self, _):
        self.stop_audio()
        self.cfg.device_index = self.device_combo.currentData()
        self.audio = AudioStream(self.cfg)
        self.start_audio()

    def on_samplerate_changed(self, val: int):
        self.stop_audio()
        self.cfg.samplerate = int(val)
        self.ring_len = int(self.cfg.seconds_in_buffer * self.cfg.samplerate)
        self.ring = np.zeros(self.ring_len, dtype=np.float32)
        self.wave_curve.setData(self.ring)
        self.audio = AudioStream(self.cfg)
        self.start_audio()

    def on_blocksize_changed(self, val: int):
        self.stop_audio()
        self.cfg.blocksize = int(val)
        self.audio = AudioStream(self.cfg)
        self.start_audio()

    def on_fft_changed(self, val: int):
        self.cfg.fft_size = int(val)

    # ---- UI Update ----
    def update_plots(self):
        # Drain audio chunks and update ring buffer
        chunks = self.audio.read_all_chunks()
        if chunks:
            total = sum(len(c) for c in chunks)
            if total >= len(self.ring):
                # If a long pause occurred, keep only the most recent samples
                new_ring = np.concatenate(chunks)[-len(self.ring):]
                self.ring[:] = new_ring
            else:
                # Shift left and append new data
                self.ring = np.roll(self.ring, -total)
                pos = len(self.ring) - total
                i = 0
                for c in chunks:
                    n = len(c)
                    self.ring[pos + i:pos + i + n] = c
                    i += n
            self.wave_curve.setData(self.ring)

        # Compute FFT on the most recent slice
        n = min(self.cfg.fft_size, len(self.ring))
        if n >= 1024:
            window = np.hanning(n).astype(np.float32)
            segment = self.ring[-n:] * window
            spec = np.fft.rfft(segment)
            mag = np.abs(spec) / (n / 2.0)
            # Avoid log(0)
            mag = np.maximum(mag, 1e-12)
            mag_db = 20 * np.log10(mag)
            freqs = np.fft.rfftfreq(n, d=1.0 / self.cfg.samplerate)
            self.fft_curve.setData(freqs, mag_db)

    # ---- Clean up ----
    def closeEvent(self, event):
        self.stop_audio()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
