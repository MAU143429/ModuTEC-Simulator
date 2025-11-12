# core/algorithms/FSK.py — BFSK por bloques reutilizando la digitalización de ASK
import numpy as np
import math

_TWO_PI = 2.0 * np.pi

# ---------------------------------------------------------------------
# Utilidades compartidas (idéntico enfoque a ASK, pero aquí local)
# ---------------------------------------------------------------------
def _alpha_from_fc(fc_hz: float, Fs: float) -> float:
    fc_hz = max(1.0, float(fc_hz)); Fs = float(Fs)
    return 1.0 - math.exp(-2.0 * math.pi * fc_hz / Fs)

def _nrz_from_abs_symbol_locked_fsk(x: np.ndarray, st: dict, tau_blk: float):
    """
    MISMA DIGITALIZACIÓN QUE ASK:
      - Trabaja sobre |x| para robustez
      - 'spb' constante (reloj de símbolo)
      - histéresis relativa alrededor de tau_blk
      - debounce por símbolos
      - conserva acumuladores entre bloques
    Devuelve bits NRZ por MUESTRA (0/1) y actualiza 'st' (acumuladores).
    """
    x = x.astype(np.float32, copy=False)
    spb   = int(st["spb"])
    band  = float(st["hyst_frac"]) * float(tau_blk)
    since = int(st["since_toggle_mod"])

    sym_sum = float(st["mod_sym_sum"])
    sym_cnt = int(st["mod_sym_cnt"])
    bit     = int(st["mod_bit"])

    bits = np.empty_like(x, dtype=np.uint8)

    for i, v in enumerate(x):
        av = v if v >= 0 else -v
        sym_sum += float(av)
        sym_cnt += 1
        bits[i] = bit
        since  += 1

        if sym_cnt >= spb:
            m = sym_sum / float(sym_cnt)
            up   = tau_blk + band
            down = tau_blk - band

            if m >= up and since >= st["debounce_syms"]:
                bit = 1; since = 0
            elif m <= down and since >= st["debounce_syms"]:
                bit = 0; since = 0
            # si m cae en (down, up) mantiene el bit previo

            sym_sum = 0.0
            sym_cnt = 0

    st["mod_sym_sum"] = float(sym_sum)
    st["mod_sym_cnt"] = int(sym_cnt)
    st["mod_bit"]     = int(bit)
    st["since_toggle_mod"] = int(since)
    return bits, st


# ---------------------------------------------------------------------
# PREPARE STATE
# ---------------------------------------------------------------------
def bfsk_prepare_state(first_chunk: np.ndarray,
                       Fs: float,
                       f_high: float,  # bit = 1
                       f_low: float,   # bit = 0
                       Ac: float,
                       bitrate: float) -> dict:
    """
    Estado persistente. Reusa EXACTAMENTE el esquema de digitalización de ASK:
      - spb (= Fs/Rb, min 2)
      - umbral por bloque (percentil de |x|)
      - histéresis y debounce por símbolo
      - continuidad de fase (CPFSK)
    La demod usa correlación I/Q + LPF 1 polo y decide por símbolo.
    """
    st = {}
    Fs = float(Fs); Rb = float(max(1.0, bitrate))
    st["Fs"] = Fs
    st["f1"] = float(f_high)
    st["f0"] = float(f_low)
    st["Ac"] = float(Ac)
    st["Rb"] = Rb

    # reloj de símbolo
    spb = int(round(Fs / Rb))
    st["spb"] = max(2, spb)

    # --- parámetros de digitalización (idénticos a ASK) ---
    st["thr_percentile"] = 70.0   # percentil aplicado sobre |x| de CADA bloque
    st["hyst_frac"]      = 0.10   # banda de histéresis como fracción de tau_blk
    st["debounce_syms"]  = 1      # símbolos mínimos entre toggles

    # acumuladores por símbolo (MOD)
    st["mod_sym_sum"] = 0.0
    st["mod_sym_cnt"] = 0
    st["mod_bit"]     = 0
    st["since_toggle_mod"] = 1e9
    st["last_tau_blk"] = 0.0

    # continuidad de portadora (CPFSK)
    st["phase"] = 0.0

    # --- DEMOD: correlación I/Q + LPF por muestra ---
    # fc_env ≈ 0.25 * Rb (suaviza unas 4 muestras por símbolo en promedio)
    env_fc = max(1.0, 0.25 * Rb)
    st["iq_alpha"] = _alpha_from_fc(env_fc, Fs)

    st["i0"] = 0.0; st["q0"] = 0.0
    st["i1"] = 0.0; st["q1"] = 0.0

    # acumuladores para decisión por símbolo (sobre y = e1 - e0)
    st["dem_sym_sum"] = 0.0
    st["dem_sym_cnt"] = 0
    st["dem_prev_bit"] = 0

    return st


# ---------------------------------------------------------------------
# MOD (audio -> bits NRZ usando el MISMO enfoque de ASK) + CPFSK
# ---------------------------------------------------------------------
def bfsk_modulate_block(x_chunk: np.ndarray, state: dict):
    """
    1) Calcula tau_blk = percentil(|x|, thr_percentile)  [idéntico a ASK]
    2) Digitaliza por símbolo (|x|) con histéresis y debounce [idéntico a ASK]
    3) CPFSK continua: f1 para bit=1, f0 para bit=0, con fase persistente
    Devuelve: s_mod, bits_nrz (uint8), state, stats{'spb','tau_blk'}
    """
    Fs = float(state["Fs"])
    f0 = float(state["f0"])
    f1 = float(state["f1"])
    Ac = float(state["Ac"])
    spb = int(state["spb"])

    # --- 1) Umbral por bloque (robusto)
    p = float(state.get("thr_percentile", 70.0))
    absx = np.abs(x_chunk.astype(np.float32, copy=False))
    tau_blk = float(np.percentile(absx, p))
    tau_blk = max(tau_blk, 1e-9)
    state["last_tau_blk"] = tau_blk

    # --- 2) NRZ por símbolo (sobre |x|) — exactamente como ASK
    bits_nrz, state = _nrz_from_abs_symbol_locked_fsk(x_chunk, state, tau_blk)  # (0/1 por muestra)

    # --- 3) CPFSK continua
    fi = np.where(bits_nrz > 0, f1, f0).astype(np.float64)
    dphi = (_TWO_PI * fi) / Fs
    phi0 = float(state.get("phase", 0.0))
    phi = phi0 + np.cumsum(dphi)
    s_mod = (Ac * np.cos(phi)).astype(np.float32)
    state["phase"] = float(phi[-1] % (2.0 * np.pi)) if s_mod.size else phi0

    stats = {"spb": spb, "tau_blk": float(tau_blk)}
    return s_mod, bits_nrz.astype(np.uint8), state, stats


# ---------------------------------------------------------------------
# DEMOD (correlación I/Q + LPF) → decisión por símbolo sobre y=e1−e0
# ---------------------------------------------------------------------
def bfsk_demodulate_block(s_chunk: np.ndarray, state: dict):
    """
    - Mezcla con cos/sin en f0 y f1 por muestra
    - LPF 1 polo (alpha=iq_alpha) para I/Q
    - Energías: e0 = sqrt(I0^2+Q0^2), e1 análogo
    - y_soft = e1 - e0 (traza continua)
    - bits_hat: integrate-and-dump por símbolo sobre y_soft (signo)
    - y_plot: mapea bits_hat a [-0.1, +0.1] (para UI)
    """
    s = s_chunk.astype(np.float64, copy=False)
    Fs = float(state["Fs"])
    f0 = float(state["f0"])
    f1 = float(state["f1"])
    spb = int(state["spb"])
    alpha = float(state.get("iq_alpha", 0.25))

    n = len(s)
    if n == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.uint8), state

    # 1) Correlación I/Q + LPF por muestra
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

    # 2) Energías y traza soft
    e0 = np.sqrt(y_i0 * y_i0 + y_q0 * y_q0)
    e1 = np.sqrt(y_i1 * y_i1 + y_q1 * y_q1)
    y_soft = (e1 - e0).astype(np.float32)

    # 3) Decisión por símbolo (integrate-and-dump sobre y_soft)
    sym_sum = float(state.get("dem_sym_sum", 0.0))
    sym_cnt = int(state.get("dem_sym_cnt", 0))
    prev    = int(state.get("dem_prev_bit", 0))

    bits_hat = np.empty(n, dtype=np.uint8)
    for k in range(n):
        sym_sum += float(y_soft[k])
        sym_cnt += 1
        # mantener el último valor hasta cerrar símbolo
        bits_hat[k] = prev
        if sym_cnt >= spb:
            m = sym_sum / float(sym_cnt)
            prev = 1 if m >= 0.0 else 0
            sym_sum = 0.0
            sym_cnt = 0

    state["dem_sym_sum"] = float(sym_sum)
    state["dem_sym_cnt"] = int(sym_cnt)
    state["dem_prev_bit"] = int(prev)

    # 4) Traza para UI: NRZ → [-Aplot, +Aplot]
    Aplot = 0.1
    y_plot = (bits_hat.astype(np.float32) * (2.0 * Aplot)) - Aplot

    return y_plot, bits_hat, state
