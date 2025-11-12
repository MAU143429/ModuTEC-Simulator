import queue
import numpy as np
from dataclasses import dataclass, field

@dataclass
class AppState:
    # --- Audio file and stream parameters ---
    audio_file_path: str | None = None
    sample_rate: int = 44100
    window_seconds: float = 2.0
    block_size: int = 2048

    # --- Audio data buffers and queues ---
    q: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=64))
    ring: np.ndarray = field(default_factory=lambda: np.zeros(44100*2, dtype=np.float32))
    mod_ring: np.ndarray = field(default_factory=lambda: np.zeros(44100*2, dtype=np.float32))
    demod_ring: np.ndarray = field(default_factory=lambda: np.zeros(44100*2, dtype=np.float32))

    # --- Audio playback and processing state ---
    reader_thread: object = None
    stop_reader: bool = False
    paused: bool = True
    pos_frames: int = 0
    timer_active: bool = False
    sd_stream: object = None   # TODO: INACTIVO
    sd_device: object = None   # TODO: INACTIVO
    audio_channels: int = 1
    is_running: bool = False
    needs_reset: bool = False  # TODO: NO ESTA SIRVIENDO

    # --- Estadísticas del bloque actual (ADAPTATIVAS) ---
    blk_mean: float = 0.0
    blk_rms: float = 0.0
    blk_peak: float = 0.0
    blk_fmax: float = 1000.0  # Hz estimado del contenido baseband del chunk

    # --- Escala/normalización ---
    # Usaremos modo "block": calcular estadísticas por chunk y no depender de valores globales.
    normalize_mode: str = "block"  # "block" (nuevo) | "global" (obsoleto)

    # --- Recommended modulation parameters (para placeholders/entradas UI) ---
    recommended_Fs: int = 0
    recommended_am_fc: int = 0
    recommended_am_Ac: float = 0.0
    recommended_am_mu: float = 0.0
    recommended_fm_fc: int = 0
    recommended_fm_Ac: float = 0.0
    recommended_fm_beta: float = 0.0
    recommended_ask_fc: int = 0
    recommended_ask_Ac: float = 0.0
    recommended_ask_bitrate: float = 0.0
    recommended_fsk_fc1: int = 0
    recommended_fsk_fc2: int = 0
    recommended_fsk_Ac: float = 0.0
    recommended_fsk_bitrate: float = 0.0

    # --- Modulation / Demodulation parameters ---
    modulation_enabled: bool = True
    modulation_type: str | None = None  # "AM" | "FM" | "ASK" | "FSK" | None

    # --- AM streaming state (fase/LPF persistentes entre bloques) ---
    am_initialized: bool = False
    am_fc: float | None = None
    am_mu: float = 0.8
    am_Ac: float | None = None
    am_phase: float = 0.0
    am_lp_ym1: float = 0.0

    # --- FM streaming state ---
   # --- Parámetros FM (UI o recomendados) ---
    fm_fc: float = 12000.0
    fm_Ac: float = 0.9
    fm_beta: float = 2.0

    # --- Estado FM (continuidad) ---
    fm_phase: float = 0.0
    fm_prev_df: float = 0.0          # <-- para suavizado de df entre bloques
    fm_prev_z: object | None = None
    fm_lp_ym1: float = 0.0
    fm_hpf_xm1: float = 0.0
    fm_hpf_ym1: float = 0.0
    fm_prev_tail: object | None = None
    fm_prev_raw: object | None = None

    # --- Perillas FM (por-bloque / demod) ---
    fm_hilbert_pad: int = 4096       # overlap grande para estabilidad
    fm_xfade: int = 512              # crossfade más largo
    fm_hpf_fc: float = 2.0           # HPF baja frecuencia
    fm_lpf_cut: float | None = None  # si None, FM.py calcula dinámico
    fm_demod_gain: float = 1.0       # ganancia fija opcional (visual)

    # --- Stats por bloque (para panel) ---
    fm_fmax_blk: float = 0.0
    fm_df_blk: float = 0.0
    fm_kappa_blk: float = 0.0

    # --- ASK (OOK) streaming state ---
    ask_initialized: bool = False
    ask_fc: float | None = None
    ask_Ac: float | None = None
    ask_bitrate: float = 2000.0
    ask_state: dict | None = None


    # --- FSK (BFSK) streaming state ---
    fsk_initialized: bool = False
    fsk_fc1: float | None = None
    fsk_fc2: float | None = None
    fsk_Ac:  float | None = None
    fsk_bitrate: float = 2000.0
    fsk_state: dict | None = None

    # --- Métricas NCC ---
    ncc_pairer: object = None
    chunk_seq: int = 0
    ncc_threshold: float = 70.0
