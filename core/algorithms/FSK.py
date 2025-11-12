import numpy as np
import math
from core.algorithms.ASK  import _nrz_from_abs_symbol_locked


_TWO_PI = 2.0 * np.pi

# =========================================================================================== #
#                                   Shared Utilities (ASK-like)                               #
# =========================================================================================== #

# Compute 1-pole IIR alpha from cutoff frequency and sample rate
def _alpha_from_fc(fc_hz: float, Fs: float) -> float:
    fc_hz = max(1.0, float(fc_hz)); Fs = float(Fs)
    return 1.0 - math.exp(-2.0 * math.pi * fc_hz / Fs)


# =========================================================================================== #
#                                         Prepare State                                       #
# =========================================================================================== #

# Initialize BFSK state (rates, symbol timing, thresholds, filters, and accumulators)
def bfsk_prepare_state(first_chunk: np.ndarray,
                       Fs: float,
                       f_high: float,  
                       f_low: float,   
                       Ac: float,
                       bitrate: float) -> dict:

    st = {}
    Fs = float(Fs); Rb = float(max(1.0, bitrate))
    st["Fs"] = Fs
    st["f1"] = float(f_high)
    st["f0"] = float(f_low)
    st["Ac"] = float(Ac)
    st["Rb"] = Rb

    # --- Symbol clock (samples per bit) ---
    spb = int(round(Fs / Rb))
    st["spb"] = max(2, spb)

    # --- Digitization params ---
    st["thr_percentile"] = 70.0  
    st["hyst_frac"]      = 0.10   
    st["debounce_syms"]  = 1     

     # --- Per-symbol accumulators (MOD) ---
    st["mod_sym_sum"] = 0.0
    st["mod_sym_cnt"] = 0
    st["mod_bit"]     = 0
    st["since_toggle_mod"] = 1e9
    st["last_tau_blk"] = 0.0

    # --- Carrier phase continuity (CPFSK) ---
    st["phase"] = 0.0

    # --- DEMOD: I/Q correlation + 1-pole LPF per sample ---
    env_fc = max(1.0, 0.25 * Rb)
    st["iq_alpha"] = _alpha_from_fc(env_fc, Fs)

    st["i0"] = 0.0; st["q0"] = 0.0
    st["i1"] = 0.0; st["q1"] = 0.0

    # --- Symbol decision accumulators over y = e1 - e0 ---
    st["dem_sym_sum"] = 0.0
    st["dem_sym_cnt"] = 0
    st["dem_prev_bit"] = 0

    return st

# =========================================================================================== #
#                                       BFSK Modulation                                       #
# =========================================================================================== #

def bfsk_modulate_block(x_chunk: np.ndarray, state: dict):

    Fs = float(state["Fs"])
    f0 = float(state["f0"])
    f1 = float(state["f1"])
    Ac = float(state["Ac"])
    spb = int(state["spb"])

    # --- Robust per-block percentile threshold over |x| ---
    p = float(state.get("thr_percentile", 70.0))
    absx = np.abs(x_chunk.astype(np.float32, copy=False))
    tau_blk = float(np.percentile(absx, p))
    tau_blk = max(tau_blk, 1e-9)
    state["last_tau_blk"] = tau_blk

    # --- NRZ per symbol over |x| with hysteresis + debounce ---
    bits_nrz, state = _nrz_from_abs_symbol_locked(x_chunk, state, tau_blk)  # (0/1 por muestra)

    # --- Continuous-phase FSK (CPFSK) using f1 for 1s and f0 for 0s ---
    fi = np.where(bits_nrz > 0, f1, f0).astype(np.float64)
    dphi = (_TWO_PI * fi) / Fs
    phi0 = float(state.get("phase", 0.0))
    phi = phi0 + np.cumsum(dphi)
    s_mod = (Ac * np.cos(phi)).astype(np.float32)
    state["phase"] = float(phi[-1] % (2.0 * np.pi)) if s_mod.size else phi0

    stats = {"spb": spb, "tau_blk": float(tau_blk)}
    return s_mod, bits_nrz.astype(np.uint8), state, stats


# =========================================================================================== #
#                                       BFSK Demodulation                                     #
# =========================================================================================== #

def bfsk_demodulate_block(s_chunk: np.ndarray, state: dict):

    s = s_chunk.astype(np.float64, copy=False)
    Fs = float(state["Fs"])
    f0 = float(state["f0"])
    f1 = float(state["f1"])
    spb = int(state["spb"])
    alpha = float(state.get("iq_alpha", 0.25))

    n = len(s)
    if n == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.uint8), state

    # --- I/Q correlation and 1-pole LPF per sample ---
    t = np.arange(n, dtype=np.float64) / Fs
    c0, s0 = np.cos(_TWO_PI * f0 * t), np.sin(_TWO_PI * f0 * t)
    c1, s1 = np.cos(_TWO_PI * f1 * t), np.sin(_TWO_PI * f1 * t)

    x0i, x0q = s * c0, s * (-s0)
    x1i, x1q = s * c1, s * (-s1)

    i0, q0 = float(state["i0"]), float(state["q0"])
    i1, q1 = float(state["i1"]), float(state["q1"])

    y_i0 = np.empty_like(x0i); y_q0 = np.empty_like(x0q)
    y_i1 = np.empty_like(x1i); y_q1 = np.empty_like(x1q)

    a = float(alpha)
    for k in range(n):
        i0 += a * (x0i[k] - i0);  q0 += a * (x0q[k] - q0)
        i1 += a * (x1i[k] - i1);  q1 += a * (x1q[k] - q1)
        y_i0[k], y_q0[k], y_i1[k], y_q1[k] = i0, q0, i1, q1

    state["i0"], state["q0"], state["i1"], state["q1"] = float(i0), float(q0), float(i1), float(q1)

    # --- Energy estimate and soft trace (e1 - e0) ---
    e0 = np.sqrt(y_i0 * y_i0 + y_q0 * y_q0)
    e1 = np.sqrt(y_i1 * y_i1 + y_q1 * y_q1)
    y_soft = (e1 - e0).astype(np.float32)

    # --- Symbol integrate-and-dump decision over y_soft ---
    sym_sum = float(state.get("dem_sym_sum", 0.0))
    sym_cnt = int(state.get("dem_sym_cnt", 0))
    prev    = int(state.get("dem_prev_bit", 0))

    bits_hat = np.empty(n, dtype=np.uint8)
    for k in range(n):
        sym_sum += float(y_soft[k])
        sym_cnt += 1
        
        bits_hat[k] = prev
        if sym_cnt >= spb:
            m = sym_sum / float(sym_cnt)
            prev = 1 if m >= 0.0 else 0
            sym_sum = 0.0
            sym_cnt = 0

    state["dem_sym_sum"] = float(sym_sum)
    state["dem_sym_cnt"] = int(sym_cnt)
    state["dem_prev_bit"] = int(prev)

    # --- UI trace, NRZ mapped ---
    Aplot = 0.1
    y_plot = (bits_hat.astype(np.float32) * (2.0 * Aplot)) - Aplot

    return y_plot, bits_hat, state
