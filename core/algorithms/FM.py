# core/algorithms/FM.py
import numpy as np
from scipy.signal import hilbert
from core.algorithms.AM import estimate_fmax_block, one_pole_lpf_block

_TWO_PI = 2.0 * np.pi


# ==============================================================
# Utilidades internas
# ==============================================================
def _one_pole_hpf(x: np.ndarray, Fs: float, fc: float, xm1: float, ym1: float):
    """HPF 1-polo implementado como y = x - LP(x)"""
    import math
    alpha = math.exp(-2.0 * math.pi * fc / Fs)
    y = np.empty_like(x, dtype=np.float32)
    x_f = x.astype(np.float32, copy=False)
    for i, xi in enumerate(x_f):
        ym1 = alpha * ym1 + (1.0 - alpha) * xi   # LP
        y[i] = xi - ym1                          # HP
        xm1 = xi
    return y, xm1, ym1


# ==============================================================
# MODULACIÓN FM (Adaptativa por bloque)
# ==============================================================
def fm_modulate_block(
    x: np.ndarray,
    Fs: float,
    state: dict,
    *,
    fc: float,
    Ac: float,
    beta: float,
    fmax_floor: float = 50.0,
    df_floor: float = 50.0,
    limit_peak: float = 0.99
) -> tuple[np.ndarray, dict, dict]:
    """
    FM adaptativa por bloque:
      - Calcula fmax_blk internamente con estimate_fmax_block
      - df_blk = max(df_floor, beta * max(fmax_floor, fmax_blk))
      - kappa = 2π * df_blk / Fs
      - Mantiene continuidad de fase
    """
    x = x.astype(np.float32, copy=False)

    # --- 1) Medición por bloque ---
    fmax_blk = float(estimate_fmax_block(x, Fs))
    rms_blk  = np.sqrt(np.mean(x**2)) + 1e-12     # energía real del bloque
    ref_rms  = 0.1                                # referencia (~-20 dBFS)
    f_ref    = max(fmax_floor, fmax_blk)

    # df adaptativo con energía real del bloque
    df_blk = max(df_floor, beta * f_ref * (rms_blk / ref_rms))
    kappa  = (_TWO_PI * df_blk) / float(Fs)


    # --- 2) Integrador de fase discreta ---
    ph0 = float(state.get("phase", 0.0))
    wc  = _TWO_PI * float(fc) / float(Fs)
    dphi = wc + kappa * x
    ph = ph0 + np.cumsum(dphi, dtype=np.float64).astype(np.float64)
    s_mod = (float(Ac) * np.cos(ph)).astype(np.float32)

    # --- 3) Evitar clipping ---
    peak = float(np.max(np.abs(s_mod)) + 1e-12)
    if peak > limit_peak:
        s_mod *= (limit_peak / peak)

    # --- 4) Actualizar estado y estadísticas ---
    state["phase"] = float(np.fmod(ph[-1], _TWO_PI))
    stats = {
        "fmax_blk": fmax_blk,
        "df_blk": df_blk,
        "kappa_blk": kappa
    }
    return s_mod, state, stats


# ==============================================================
# DEMODULACIÓN FM (Adaptativa por bloque)
# ==============================================================
def fm_demodulate_block(
    s: np.ndarray,
    Fs: float,
    state: dict,
    *,
    df_blk: float,
    use_overlap_hilbert: bool = True
) -> tuple[np.ndarray, dict]:
    """
    Discriminador robusto por bloque con continuidad.
    """
    s_blk = s.astype(np.float32, copy=False)

    # --- Estados y parámetros ---
    P = int(state.get("hilbert_pad", 1024))
    XFADE = int(state.get("xfade", 192))
    hpf_fc = float(state.get("hpf_fc", 0.5))
    lp_ym1 = float(state.get("lp_ym1", 0.0))
    prev_z = state.get("prev_z", None)
    prev_raw = state.get("prev_raw", None) if use_overlap_hilbert else None
    prev_tail = state.get("prev_tail", None)

    # --- 1) Señal analítica ---
    if use_overlap_hilbert:
        if prev_raw is None or len(prev_raw) < P:
            pad = np.zeros(P, dtype=np.float32)
        else:
            pad = prev_raw[-P:].astype(np.float32)
        s_ext = np.concatenate([pad, s_blk], axis=0)
        z_ext = hilbert(s_ext).astype(np.complex64)
        mag = np.abs(z_ext) + 1e-12
        z_ext = (z_ext / mag).astype(np.complex64)
        z = z_ext[-len(s_blk):]
    else:
        z = hilbert(s_blk).astype(np.complex64)
        z /= np.abs(z) + 1e-12

    # --- 2) Diferencia de fase (continuidad) ---
    if prev_z is None:
        w_rest = z[1:] * np.conj(z[:-1])
        w = np.concatenate([w_rest[:1], w_rest], axis=0)
    else:
        w0 = z[0] * np.conj(prev_z)
        w_rest = z[1:] * np.conj(z[:-1])
        w = np.concatenate([w0[None], w_rest], axis=0)

    dphi = np.arctan2(np.imag(w), np.real(w)).astype(np.float32)
    f_inst = (dphi * Fs) / (2.0 * np.pi)

    # --- 3) Clamp por MAD + HPF ---
    med = float(np.median(f_inst))
    mad = float(np.median(np.abs(f_inst - med)) + 1e-12)
    f_inst = np.clip(f_inst, med - 2.0 * mad, med + 2.0 * mad)
    f_base, hpf_xm1, hpf_ym1 = _one_pole_hpf(
        f_inst, Fs, hpf_fc,
        float(state.get("hpf_xm1", 0.0)),
        float(state.get("hpf_ym1", 0.0))
    )

    # --- 4) Normalización por df_blk + LPF salida ---
    m_est = (f_base / (df_blk + 1e-12)).astype(np.float32)
    fmax_blk_for_lpf = float(state.get("last_fmax_blk", 1000.0))
    cut = state.get("lpf_cut", None)
    if cut is None:
        cut = min(0.05 * Fs, max(1.2 * fmax_blk_for_lpf, 2000.0))

    m_lp, lp_ym1 = one_pole_lpf_block(m_est, Fs, float(cut), lp_ym1)

    # --- 5) Recentrado DC y ganancia fija ---
    m_lp = m_lp - np.mean(m_lp)        # elimina DC real sin escalar forma
    #m_lp *= 10.0                       # ganancia fija para visualización (ajustable)


    if prev_tail is not None and XFADE > 0 and len(m_lp) > XFADE:
        w_in = np.linspace(0.0, 1.0, XFADE, dtype=np.float32)
        w_out = 1.0 - w_in
        m_lp[:XFADE] = w_out * prev_tail[-XFADE:] + w_in * m_lp[:XFADE]

    # --- Persistir estados ---
    state["prev_raw"] = s_blk.copy() if use_overlap_hilbert else None
    state["prev_z"] = np.complex64(z[-1])
    state["prev_tail"] = m_lp[-XFADE:].copy() if XFADE > 0 else None
    state["lp_ym1"] = float(lp_ym1)
    state["hpf_xm1"] = float(hpf_xm1)
    state["hpf_ym1"] = float(hpf_ym1)

    return m_lp, state


# ==============================================================
# PROCESO COMPLETO POR BLOQUE
# ==============================================================
def fm_process_block(
    x: np.ndarray,
    Fs: float,
    state: dict,
    *,
    fc: float,
    Ac: float,
    beta: float
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Modula y demodula un bloque completo adaptativamente."""
    s_mod, st_mid, stats = fm_modulate_block(
        x, Fs, state, fc=fc, Ac=Ac, beta=beta
    )
    st_mid["last_fmax_blk"] = float(stats["fmax_blk"])
    s_dem, st_fin = fm_demodulate_block(
        s_mod, Fs, st_mid, df_blk=float(stats["df_blk"])
    )
    return s_mod, s_dem, st_fin, stats
