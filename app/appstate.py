import queue
import numpy as np
from dataclasses import dataclass, field


@dataclass
class AppState:
    audio_file_path: str | None = None
    sample_rate: int = 44100
    window_seconds: float = 4.0
    block_size: int = 2048


    ring: np.ndarray = field(default_factory=lambda: np.zeros(44100*4, dtype=np.float32))
    q: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=64))

    reader_thread: object = None
    stop_reader: bool = False
    paused: bool = True
    pos_frames: int = 0
    timer_active: bool = False
    sd_stream: object = None
    sd_device: object = None
    audio_channels: int = 1
    is_running: bool = False
