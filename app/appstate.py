import queue
import numpy as np
from dataclasses import dataclass, field


@dataclass
class AppState:
    
    # --- Audio file and stream parameters ---
    
    audio_file_path: str | None = None                                                           # Audio file path
    sample_rate: int = 44100                                                                     # Sample rate
    fmax: int = sample_rate/4
    window_seconds: float = 2.0                                                                  # Window (Graph) size in seconds
    block_size: int = 2048                                                                       # Block size for audio processing

    # --- Audio data buffers and queues ---
    
    q: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=64))                      # Queue for audio data blocks
    ring: np.ndarray = field(default_factory=lambda: np.zeros(44100*2, dtype=np.float32))        # Ring buffer for original audio data (2 seconds max)
    mod_ring: np.ndarray = field(default_factory=lambda: np.zeros(44100*2, dtype=np.float32))    # Ring buffer for modulated audio data (2 seconds max)
    demod_ring: np.ndarray = field(default_factory=lambda: np.zeros(44100*2, dtype=np.float32))  # Ring buffer for demodulated audio data (2 seconds max)

    # --- Audio playback and processing state ---
    
    reader_thread: object = None                                                                 # Thread for reading audio file
    stop_reader: bool = False                                                                    # Flag to stop reader thread
    paused: bool = True                                                                          # Playback paused state
    pos_frames: int = 0                                                                          # Current position in frames
    timer_active: bool = False                                                                   # Timer active state
    sd_stream: object = None #TODO INACTIVO                                                           # Sounddevice stream object
    sd_device: object = None #TODO INACTIVO                                                           # Sounddevice device info
    audio_channels: int = 1                                                                      # Number of audio channels
    is_running: bool = False                                                                     # Is the audio processing running
    needs_reset: bool = False #TODO NO ESTA SIRVIENDO                                                 # Flag to reset state
    
    
    # --- Recommended modulation parameters ---
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
    
    recommended_fsk_fc: int = 0
    recommended_fsk_Ac: float = 0.0
    recommended_fsk_bitrate: float = 0.0
    
    
    # --- Modulation / Demodulation parameters ---
    
    modulation_enabled: bool = True                                                              # Enable modulation/demodulation
    modulation_type: str | None = None                                                           # "AM" | "FM" | "ASK" | "FSK" | None                                                          # "envelope" | "coherent"
    
    # --- AM streaming state ---
    am_initialized: bool = False                                                                 # True after the first chunk
    am_fc: float | None = None                                                                   # Carrier frequency set at start
    am_mu: float = 0.8                                                                           # Modulation index set at start 
    am_Ac: float | None = None                                                                   # Carrier amplitude set at start
    am_phase: float = 0.0                                                                        # Accumulated phase (rad), for continuity
    am_xscale: float | None = None                                                               # Fixed signal scale (robust peak from first chunk)
