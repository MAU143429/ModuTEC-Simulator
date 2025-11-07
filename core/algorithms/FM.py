# core/algorithms/FM.py
import numpy as np
from scipy.signal import hilbert

# Reutilizamos utilidades desde AM para no duplicar
from core.algorithms.AM import robust_peak, estimate_fmax_fft, one_pole_lpf_block

_TWO_PI = 2.0 * np.pi

def fm_prepare_state(first_chunk: np.ndarray, Fs: float,
                     fc: float | None, Ac: float | None, beta: float | None) -> dict:
    """
    Decide parámetros fijos de FM a partir del primer bloque y congela escala.
    - xscale: pico robusto (para normalizar estable en [-1,1])
    - fc: si no viene, lo fijamos en función del Fs
    - Ac: amplitud de la portadora
    - beta: índice de modulación; si no viene, preparamos uno por defecto
    - Δf = beta * fmax (fmax estimada del primer chunk)
    - kappa = 2π * Δf / Fs (ganancia de fase por-sample que integra el mensaje)
    """
    xscale = robust_peak(first_chunk) + 1e-12

    if fc is None:
        # mantener fc segura y bien por debajo de Nyquist
        fmax = estimate_fmax_fft(first_chunk, Fs)
        fc = min(0.25 * Fs, max(200.0, 10.0 * fmax))

    if Ac is None:
        Ac = 0.7  # valor estable por defecto (puedes exponerlo si quieres)

    # Índice de modulación (si no te lo dan)
    if beta is None or beta <= 0:
        # algo razonable para empezar (wideband light)
        beta = 2.0

    # fmax para calcular Δf
    fmax = estimate_fmax_fft(first_chunk, Fs)
    df = max(75.0, beta * max(75.0, fmax))  # Δf mínimo de trabajo
    #df = max(50.0, beta * max(50.0, fmax))

    # kappa implementa la integral discreta (acumularemos fase)
    kappa = _TWO_PI * df / float(Fs)

    state = {
        "fc": float(fc),
        "Ac": float(Ac),
        "beta": float(beta),
        "xscale": float(xscale),
        "phase": 0.0,                 # fase acumulada de la portadora
        "kappa": float(kappa),        # ganancia de fase por muestra
        "fmax": float(fmax),          # por si lo ocupamos en demod LPF
        "phase_unwrap_prev": 0.0,     # para demod por diferenciación de fase
        "lp_ym1": 0.0,                # estado de LPF 1 polo en demod
    }
    return state


def fm_modulate_block(x: np.ndarray, Fs: float, state: dict) -> tuple[np.ndarray, dict]:
    """
    FM discreta: fase[n] = fase[n-1] + 2πfc/Fs + kappa * m_norm[n]
    donde m_norm in [-1,1] ~ integral discreta (por acumulación en fase).
    """
    fc     = float(state["fc"])
    Ac     = float(state["Ac"])
    ph0    = float(state["phase"])
    kappa  = float(state["kappa"])
    xscale = float(state["xscale"])

    # normalización fija por-run
    x_norm = (x.astype(np.float32) / xscale).astype(np.float32)
    x_norm = np.clip(x_norm, -1.0, 1.0)

    L   = len(x_norm)
    wc  = _TWO_PI * fc / float(Fs)

    # incremento de fase por muestra: portadora + “integral” del mensaje
    # (la suma de kappa*x_norm[n] en la acumulación realiza la integral discreta)
    n = np.arange(L, dtype=np.float32)
    # para evitar bucles Python, acumulamos vectorialmente
    dphi = wc + kappa * x_norm
    ph   = ph0 + np.cumsum(dphi, dtype=np.float64).astype(np.float64)
    s    = (Ac * np.cos(ph)).astype(np.float32)

    # evita clipping best-effort
    peak = float(np.max(np.abs(s)) + 1e-12)
    if peak > 0.99:
        s *= (0.99 / peak)

    # guardar última fase (mod 2π para estabilidad numérica)
    ph1 = float(np.fmod(ph[-1], _TWO_PI))

    new_state = dict(state)
    new_state["phase"] = ph1
    return s, new_state


def _one_pole_hpf_block(x, Fs, fc, xm1=0.0, ym1=0.0):
    # HPF: y[n] = a*(y[n-1] + x[n] - x[n-1])
    # a = 1 / (1 + 2π fc / Fs)
    a = 1.0 / (1.0 + (_TWO_PI*fc)/Fs)
    y = np.empty_like(x, dtype=np.float32)
    prev_x = xm1
    prev_y = ym1
    for i, xi in enumerate(x):
        yi = a*(prev_y + xi - prev_x)
        y[i] = yi
        prev_y = yi
        prev_x = xi
    return y, float(prev_x), float(prev_y)

def fm_demodulate_block(s: np.ndarray, Fs: float, state: dict):
    """
    FM demod por discriminador robusta en streaming:
    - Overlap-save para Hilbert (evita artefactos de borde)
    - Continuidad exacta en la 1ª diferencia (usa prev_z)
    - De-glitch por MAD + HPF muy bajo
    - LPF 1 polo a banda de audio
    - Crossfade corto con el bloque previo
    """

    # ---------- parámetros de streaming ----------
    P = int(state.get("hilbert_pad", 1024))    # overlap para Hilbert
    XFADE = int(state.get("xfade", 192))       # crossfade en muestras

    kappa   = float(state["kappa"])
    fmax    = float(state.get("fmax", 4000.0))
    lp_ym1  = float(state.get("lp_ym1", 0.0))
    prev_raw = state.get("prev_raw", None)
    prev_z   = state.get("prev_z", None)
    prev_tail = state.get("prev_tail", None)  # cola del demod anterior para xfade

    # ---------- overlap-save antes del Hilbert ----------
    if prev_raw is None or len(prev_raw) < P:
        pad = np.zeros(P, dtype=np.float32)
    else:
        pad = prev_raw[-P:].astype(np.float32)

    s_blk = s.astype(np.float32)
    s_ext = np.concatenate([pad, s_blk], axis=0)

    # Analítica y AGC suave (módulo ≈ 1) en el bloque extendido
    z_ext = hilbert(s_ext).astype(np.complex64)
    mag   = np.abs(z_ext) + 1e-12
    z_ext = (z_ext / mag).astype(np.complex64)

    # Nos quedamos con la parte del bloque actual
    z = z_ext[-len(s_blk):]

    # ---------- discriminador con continuidad ----------
    if prev_z is None:
        # continuidad “suave” si es el 1er bloque
        w_rest = z[1:] * np.conj(z[:-1])
        w = np.concatenate([w_rest[:1], w_rest], axis=0)
    else:
        w0 = z[0] * np.conj(prev_z)
        w_rest = z[1:] * np.conj(z[:-1])
        w = np.concatenate([w0[None], w_rest], axis=0)

    dphi   = np.arctan2(np.imag(w), np.real(w)).astype(np.float32)  # rad/muestra
    f_inst = (dphi * Fs) / (2.0*np.pi)                               # Hz

    # ---------- de-glitch + HPF muy bajo ----------
    med = np.median(f_inst)
    mad = np.median(np.abs(f_inst - med)) + 1e-12
    lim = 2.0 * mad
    f_inst = np.clip(f_inst, med - lim, med + lim)
    
    # HPF 0.5 Hz (ajustable en state si quieres)
    def _one_pole_hpf(x, Fs, fc, xm1, ym1):
        # a partir de tu implementación equivalente
        import math
        alpha = math.exp(-2.0*math.pi*fc/Fs)
        y = np.empty_like(x, dtype=np.float32)
        for i, xi in enumerate(x.astype(np.float32)):
            # filtro pasa-altas simple mediante resta LP
            ym1 = alpha*ym1 + (1.0-alpha)*xi
            y[i] = xi - ym1
            xm1 = xi
        return y, xm1, ym1

    hpf_fc = float(state.get("hpf_fc", 0.5))
    f_base, hpf_xm1, hpf_ym1 = _one_pole_hpf(f_inst, Fs, hpf_fc,
                                             state.get("hpf_xm1", 0.0),
                                             state.get("hpf_ym1", 0.0))

    # ---------- normalización y LPF a banda de audio ----------
    df = (kappa * Fs) / (2.0*np.pi)
    m_est = (f_base / (df + 1e-12)).astype(np.float32)

    # LPF 1 polo ~ ligeramente por encima de fmax
    cut = float(state.get("lpf_cut", max(1.1*fmax, fmax + 200.0)))
    m_lp, lp_ym1 = one_pole_lpf_block(m_est, Fs, cut, lp_ym1)

    m_lp -= np.mean(m_lp)
    peak = np.max(np.abs(m_lp)) + 1e-12
    m_lp = (m_lp / peak).astype(np.float32)

    # ---------- crossfade con el bloque previo ----------
    if prev_tail is not None and XFADE > 0 and len(m_lp) > XFADE:
        # mezcla lineal de XFADE muestras iniciales
        w_in  = np.linspace(0.0, 1.0, XFADE, dtype=np.float32)
        w_out = 1.0 - w_in
        m_lp[:XFADE] = w_out * prev_tail[-XFADE:] + w_in * m_lp[:XFADE]

    # ---------- actualizar estados ----------
    state["prev_raw"] = s_blk.copy()
    state["prev_z"]   = np.complex64(z[-1])
    state["prev_tail"]= m_lp[-XFADE:].copy() if XFADE > 0 else None
    state["lp_ym1"]   = float(lp_ym1)
    state["hpf_xm1"]  = float(hpf_xm1)
    state["hpf_ym1"]  = float(hpf_ym1)

    return m_lp, state
