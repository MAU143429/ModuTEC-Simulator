import numpy as np

_TWO_PI = 2.0*np.pi

def am_prepare_state(first_chunk: np.ndarray):
    
    xscale = robust_peak(first_chunk) + 1e-12

    return float(xscale)
    

def am_modulate_block(x: np.ndarray, Fs: float, state: dict) -> tuple[np.ndarray, dict]:
    """
    Modula un bloque 'x' usando parámetros fijos y fase continua en 'state'.
    Devuelve (s_mod, state_actualizado).
    """
    fc   = state['fc']
    mu   = state['mu']
    Ac   = state['Ac']
    ph0  = state['phase']
    scl  = state['xscale']

    # Normalización estable (congelada desde el 1er chunk)
    x_norm = (x.astype(np.float32) / scl).astype(np.float32)
    x_norm = np.clip(x_norm, -1.0, 1.0)

    L   = len(x_norm)
    omg = _TWO_PI * fc / float(Fs)

    # Portadora con fase continua
    n   = np.arange(L, dtype=np.float32)
    ph  = ph0 + omg*n
    carrier = np.cos(ph).astype(np.float32)

    s = (Ac * (1.0 + mu * x_norm) * carrier).astype(np.float32)

    # Limit suave por seguridad
    mod_peak = float(np.max(np.abs(s)) + 1e-12)
    if mod_peak > 0.99:
        s *= (0.99 / mod_peak)

    # Actualiza fase acumulada (mantén en [0, 2π) para estabilidad numérica)
    ph1 = ph0 + omg*L
    ph1 = float(np.fmod(ph1, _TWO_PI))

    state_updated = dict(state)
    state_updated['phase'] = ph1
    return s, state_updated

def am_demodulate_block(s: np.ndarray, Fs: float, state: dict, smooth_frac: float = 0.15) -> np.ndarray:
    
    """
    Demodulación por envolvente (rectificación + LPF).
    Usa filtro de 1 polo (doble pasada opcional) con estado persistente.
    """
    fc = float(state["fc"])
    mu = float(np.clip(state["mu"], 0.0, 1.0))
    Ac = float(state["Ac"])
    fmax = float(state.get("fmax", 1000.0))
    y_prev = float(state.get("lp_ym1", 0.0))

    # 1. Rectificación (envolvente)
    env = np.abs(s).astype(np.float32)

    # 2. LPF con corte bajo para eliminar portadora
    cut = min(0.25 * Fs, max(1.1 * fmax, fmax + 100.0))
    env_lp, y_last = one_pole_lpf_block(env, Fs, cut, y_prev)
    # Doble pasada para limpieza extra
    env_lp2, y_last2 = one_pole_lpf_block(env_lp, Fs, cut, y_last)
    state["lp_ym1"] = y_last2
    base = env_lp2

    # 3. Escalado físico y centrado
    demod = ((base / (Ac + 1e-12)) - 1.0) / (mu + 1e-12)
    demod = demod - np.mean(demod)
    demod = demod.astype(np.float32)

    # 4. Normalización visual [-1, 1]
    demod = demod / (np.max(np.abs(demod)) + 1e-12)

    return demod

# ==========================
# Utilidades
# ==========================
def robust_peak(x):
    return float(np.percentile(np.abs(x), 99.9))

def estimate_fmax_fft(x, Fs):
    N = min(len(x), 1 << 18)
    X = np.fft.rfft(x[:N])
    mag2 = np.abs(X)**2
    freqs = np.fft.rfftfreq(N, d=1.0/Fs)
    c = np.cumsum(mag2)
    cp = c / (c[-1] if c[-1] > 0 else 1.0)
    idx = np.searchsorted(cp, 0.99)
    return float(freqs[min(idx, len(freqs)-1)])

def moving_average(x, M):
    if M <= 1:
        return x.astype(np.float32, copy=True)
    kernel = np.ones(M, dtype=np.float32) / float(M)
    return np.convolve(x.astype(np.float32), kernel, mode='same')

def one_pole_lpf_block(x: np.ndarray, Fs: float, fc_hz: float, y_prev: float) -> tuple[np.ndarray, float]:
    fc = max(1.0, float(fc_hz))
    alpha = 1.0 - np.exp(-2.0*np.pi*fc/float(Fs))
    y = np.empty_like(x, dtype=np.float32)
    y_last = float(y_prev)
    for i in range(len(x)):
        y_last = y_last + alpha*(float(x[i]) - y_last)
        y[i] = y_last
    return y, y_last