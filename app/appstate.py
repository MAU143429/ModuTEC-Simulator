import queue
import numpy as np
from dataclasses import dataclass, field


@dataclass
class AppState:
    audio_file_path: str | None = None
    sample_rate: int = 44100
    window_seconds: float = 2.0
    block_size: int = 2048


    ring: np.ndarray = field(default_factory=lambda: np.zeros(44100*2, dtype=np.float32))
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
    needs_reset: bool = False
    
    
    # --- Modulación en tiempo real (plot 2) ---
    modulation_enabled: bool = True       # ON por defecto (si mod_type es AM)
    modulation_type: str = "AM"           # AM | FM | ASK | FSK (por ahora AM)
    mod_ring: np.ndarray = field(default_factory=lambda: np.zeros(44100*2, dtype=np.float32))
    demod_ring: np.ndarray = field(default_factory=lambda: np.zeros(44100*2, dtype=np.float32))
    demod_method: str = "envelope"            # "envelope" | "coherent"
    
    # --- AM streaming state (persistente para toda la sesión) ---
    am_initialized: bool = False     # True después del 1er chunk
    am_fc: float | None = None       # portadora fijada al iniciar
    am_mu: float = 0.8               # índice de modulación fijado al iniciar (o el que uses)
    am_Ac: float | None = None       # amplitud de portadora fijada al iniciar
    am_phase: float = 0.0            # fase acumulada (rad), para continuidad
    am_xscale: float | None = None   # escala fija de la señal (pico robusto del 1er chunk)

        