import math
import numpy as np
from scipy.signal import hilbert
from core.algorithms.AM import estimate_fmax_block, one_pole_lpf_block

_TWO_PI = 2.0 * np.pi


# =========================================================================================== #
#                                    FM Internal Utilities                                    #
# =========================================================================================== #

# Apply a simple one-pole HPF via y = x − LP(x); returns y and updated states
def _one_pole_hpf(x: np.ndarray, Fs: float, fc: float, xm1: float, ym1: float):
    alpha = math.exp(-2.0 * math.pi * fc / Fs)
    y = np.empty_like(x, dtype=np.float32)
    x_f = x.astype(np.float32, copy=False)
    for i, xi in enumerate(x_f):
        ym1 = alpha * ym1 + (1.0 - alpha) * xi  
        y[i] = xi - ym1                         
        xm1 = xi
    return y, xm1, ym1


# =========================================================================================== #
#                                        FM Modulation                                        #
# =========================================================================================== #

# Modulate one block of FM with adaptive frequency deviation and phase continuity
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

    x = x.astype(np.float32, copy=False)

    # --- Block measurement ---
    fmax_blk = float(estimate_fmax_block(x, Fs))
    rms_blk  = np.sqrt(np.mean(x**2)) + 1e-12    
    ref_rms  = 0.1                                
    f_ref    = max(fmax_floor, fmax_blk)

    # --- Adaptive deviation and kappa ---
    df_blk = max(df_floor, beta * f_ref * (rms_blk / ref_rms))
    kappa  = (_TWO_PI * df_blk) / float(Fs)


    # --- Discrete phase integrator with continuity ---
    ph0 = float(state.get("phase", 0.0))
    wc  = _TWO_PI * float(fc) / float(Fs)
    dphi = wc + kappa * x
    ph = ph0 + np.cumsum(dphi, dtype=np.float64).astype(np.float64)
    s_mod = (float(Ac) * np.cos(ph)).astype(np.float32)


    # --- Peak limiting to avoid clipping ---
    peak = float(np.max(np.abs(s_mod)) + 1e-12)
    if peak > limit_peak:
        s_mod *= (limit_peak / peak)
        

    # --- Update state and expose per-block stats ---
    state["phase"] = float(np.fmod(ph[-1], _TWO_PI))
    stats = {
        "fmax_blk": fmax_blk,
        "df_blk": df_blk,
        "kappa_blk": kappa
    }
    return s_mod, state, stats


# =========================================================================================== #
#                                        FM Demodulation                                      #
# =========================================================================================== #

# Demodulate one block using analytic signal, phase diff, HPF+LPF; keeps continuity
def fm_demodulate_block(
    s: np.ndarray,
    Fs: float,
    state: dict,
    *,
    df_blk: float,
    use_overlap_hilbert: bool = True
) -> tuple[np.ndarray, dict]:

    s_blk = s.astype(np.float32, copy=False)

    # --- Load persistent states and params ---
    
    P = int(state.get("hilbert_pad", 1024))
    XFADE = int(state.get("xfade", 192))
    hpf_fc = float(state.get("hpf_fc", 0.5))
    lp_ym1 = float(state.get("lp_ym1", 0.0))
    prev_z = state.get("prev_z", None)
    prev_raw = state.get("prev_raw", None) if use_overlap_hilbert else None
    prev_tail = state.get("prev_tail", None)
    

    # --- Analytic signal with optional overlap pad ---
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


    # --- Phase difference with continuity to previous block ---
    if prev_z is None:
        w_rest = z[1:] * np.conj(z[:-1])
        w = np.concatenate([w_rest[:1], w_rest], axis=0)
    else:
        w0 = z[0] * np.conj(prev_z)
        w_rest = z[1:] * np.conj(z[:-1])
        w = np.concatenate([w0[None], w_rest], axis=0)

    dphi = np.arctan2(np.imag(w), np.real(w)).astype(np.float32)
    f_inst = (dphi * Fs) / (2.0 * np.pi)
    
    
    # --- Robust clamp median and high-pass baseband ---
    med = float(np.median(f_inst))
    mad = float(np.median(np.abs(f_inst - med)) + 1e-12)
    f_inst = np.clip(f_inst, med - 2.0 * mad, med + 2.0 * mad)
    f_base, hpf_xm1, hpf_ym1 = _one_pole_hpf(
        f_inst, Fs, hpf_fc,
        float(state.get("hpf_xm1", 0.0)),
        float(state.get("hpf_ym1", 0.0))
    )

    # --- Normalize by df_blk and low-pass smooth output ---
    m_est = (f_base / (df_blk + 1e-12)).astype(np.float32)
    fmax_blk_for_lpf = float(state.get("last_fmax_blk", 1000.0))
    cut = state.get("lpf_cut", None)
    if cut is None:
        
        cut = max(1.1 * fmax_blk_for_lpf, fmax_blk_for_lpf + 200.0)
        cut = min(cut, 0.10 * Fs)
        cut = max(cut, 40.0)
    m_lp, lp_ym1 = one_pole_lpf_block(m_est, Fs, float(cut), lp_ym1)
    

    # --- Remove residual DC only ---
    m_lp = (m_lp - float(np.mean(m_lp))).astype(np.float32)

    # --- Crossfade with previous tail for visual continuity ---
    if prev_tail is not None and XFADE > 0 and len(m_lp) > XFADE:
        w_in = np.linspace(0.0, 1.0, XFADE, dtype=np.float32)
        w_out = 1.0 - w_in
        m_lp[:XFADE] = w_out * prev_tail[-XFADE:] + w_in * m_lp[:XFADE]

    # --- Persist updated states for next block ---
    state["prev_raw"] = s_blk.copy() if use_overlap_hilbert else None
    state["prev_z"] = np.complex64(z[-1])
    state["prev_tail"] = m_lp[-XFADE:].copy() if XFADE > 0 else None
    state["lp_ym1"] = float(lp_ym1)
    state["hpf_xm1"] = float(hpf_xm1)
    state["hpf_ym1"] = float(hpf_ym1)

    return m_lp, state


# =========================================================================================== #
#                                    Full FM Block Process                                    #
# =========================================================================================== #

# Run full FM step for one block (modulate then demodulate) with adaptive params
def fm_process_block(
    x: np.ndarray,
    Fs: float,
    state: dict,
    *,
    fc: float,
    Ac: float,
    beta: float
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    
    # --- Modulate with adaptive df and phase continuity ---
    s_mod, st_mid, stats = fm_modulate_block(
        x, Fs, state, fc=fc, Ac=Ac, beta=beta
    )
    st_mid["last_fmax_blk"] = float(stats["fmax_blk"])
    
    # --- Demodulate using current df and saved states ---
    s_dem, st_fin = fm_demodulate_block(
        s_mod, Fs, st_mid, df_blk=float(stats["df_blk"])
    )
    return s_mod, s_dem, st_fin, stats
