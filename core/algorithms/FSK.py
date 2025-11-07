# core/algorithms/FSK.py — BFSK (dos tonos) con mapeo audio→bits y continuidad por bloques
import numpy as np
_TWO_PI = 2.0*np.pi

# ---------- Utilidades internas ----------

def _robust_peak(x: np.ndarray) -> float:
    return float(np.percentile(np.abs(x), 99.0) + 1e-12)

def _alpha_from_fc(fc_hz: float, Fs: float) -> float:
    fc_hz = max(1.0, float(fc_hz))
    return 1.0 - np.exp(-2.0*np.pi*fc_hz / Fs)

def _schmitt_sample(v: float, state_val: float, low: float, high: float) -> float:
    # Histéresis binaria para binarizar PCM (igual patrón que ASK)
    if state_val <= 0.5:
        return 1.0 if v >= high else 0.0
    else:
        return 0.0 if v <= low else 1.0

# ---------- Estado y preparación ----------

def bfsk_prepare_state(first_chunk: np.ndarray,
                       Fs: float,
                       f_low: float,
                       f_high: float,
                       Ac: float,
                       bitrate: float):
    """
    Prepara parámetros fijos y estado persistente para BFSK.
    Entradas del usuario: Fs, f_low (0), f_high (1), Ac, bitrate.
    - spb = Fs/bitrate (mínimo 2)
    - xscale desde primer bloque (pico robusto)
    - Histéresis adaptativa en audio normalizado (como ASK)
    - Demod: correlación I/Q + LPF 1 polo (corte ≈ Rb/4)
    """
    x = first_chunk.astype(np.float64, copy=False)
    xscale = _robust_peak(x)
    spb = max(2, int(round(Fs / max(1.0, float(bitrate)))))

    # Umbrales sobre el primer bloque normalizado (como ASK)
    xn = x / xscale if xscale > 0 else x
    p95 = float(np.percentile(np.abs(xn), 95.0)) if len(xn) else 1.0
    thr_high = 0.35 * p95
    thr_low  = 0.15 * p95

    # LPF para I/Q en demod (suave respecto al bitrate)
    env_fc = max(1.0, 0.25 * bitrate)
    env_alpha = _alpha_from_fc(env_fc, Fs)

    state = {
        # Parámetros fijos
        "Fs": float(Fs),
        "f0": float(f_low),     # tono para bit 0
        "f1": float(f_high),    # tono para bit 1
        "Ac": float(Ac),
        "bitrate": float(bitrate),
        "spb": int(spb),
        "xscale": float(xscale),

        # Modulación (binarización + reloj símbolo + fase continua CPFSK)
        "mod_thr_low":  float(thr_low),
        "mod_thr_high": float(thr_high),
        "mod_state": 0.0,          # estado Schmitt (0/1)
        "mod_one_count": 0,
        "mod_sample_in_sym": 0,
        "mod_curr_bit": 0.0,
        "phase": 0.0,              # fase acumulada (única) para continuidad CPFSK

        # Demod I/Q + LPF + decisión con histéresis en la traza "soft" y=e1-e0
        "alpha": float(env_alpha),
        "i0": 0.0, "q0": 0.0,      # acumuladores LPF para tono f0
        "i1": 0.0, "q1": 0.0,      # acumuladores LPF para tono f1
        "demod_state": 0.0,        # estado Schmitt sobre y=e1-e0
        "hyst": 0.15,              # ±15% de histéresis sobre umbral 0
        "run_len": 0,              # deglitch: longitud de racha
    }
    return state

# ---------- Modulación ----------

def bfsk_modulate_block(x_chunk: np.ndarray, state: dict):
    """
    Modulación BFSK desde PCM (audio→bits→CPFSK):
      1) Normaliza por xscale
      2) Binariza por histéresis (Schmitt)
      3) Decisión por símbolo (mayoría cada spb) con continuidad entre bloques
      4) CPFSK: fase integra frecuencia por muestra (f1 si bit=1, f0 si bit=0)
    Devuelve: s (modulada), bits_nrz_por_muestra (0/1)
    """
    Fs, f0, f1, Ac, spb = state["Fs"], state["f0"], state["f1"], state["Ac"], state["spb"]
    xscale = state["xscale"]
    thr_low, thr_high = state["mod_thr_low"], state["mod_thr_high"]

    mod_state = state["mod_state"]
    ones = state["mod_one_count"]
    k = state["mod_sample_in_sym"]
    curr_bit = state["mod_curr_bit"]
    phase = state["phase"]

    x = x_chunk.astype(np.float64, copy=False) / xscale
    n = len(x)
    bits_nrz = np.empty(n, dtype=np.float64)

    # Generar bits NRZ por muestra con reloj de símbolo continuo
    for i in range(n):
        b_sample = _schmitt_sample(x[i], mod_state, thr_low, thr_high)
        mod_state = b_sample
        ones += 1 if b_sample >= 0.5 else 0
        k += 1

        # Mantener el bit de salida hasta cerrar símbolo
        bits_nrz[i] = curr_bit

        if k >= spb:
            curr_bit = 1.0 if (ones >= (spb - ones)) else 0.0
            ones = 0
            k = 0

    # CPFSK: integrar fase según frecuencia instantánea
    fi = np.where(bits_nrz > 0.5, f1, f0)
    # φ[n] = φ[n-1] + 2π*fi[n]/Fs
    ph = phase + _TWO_PI * np.cumsum(fi)/Fs
    s = Ac * np.cos(ph)
    phase = float(np.mod(ph[-1], _TWO_PI)) if n else phase

    # Guardar estado
    state["mod_state"] = float(mod_state)
    state["mod_one_count"] = int(ones)
    state["mod_sample_in_sym"] = int(k)
    state["mod_curr_bit"] = float(curr_bit)
    state["phase"] = float(phase)

    return s, bits_nrz

# ---------- Demodulación ----------

def bfsk_demodulate_block(s_chunk: np.ndarray, state: dict):
    """
    Demod BFSK:
      - Mezcla con cos/sin en f0 y f1 por muestra
      - LPF 1-polo en I/Q
      - Energía e0 = sqrt(I0^2+Q0^2), e1 análogo
      - y = e1 - e0 (soft)
      - bits_hat reconstruido cada símbolo (NRZ 0/1)
    Devuelve:
      y_soft: traza continua (útil para comparar)
      bits_hat_nrz: señal digital reconstruida (0/1 por muestra)
    """
    s = s_chunk.astype(np.float64, copy=False)
    Fs, f0, f1, alpha, spb = (
        state["Fs"], state["f0"], state["f1"], state["alpha"], state["spb"]
    )
    n = len(s)
    if n == 0:
        return np.zeros(0), np.zeros(0)

    # 1) Correlación I/Q
    t = np.arange(n) / Fs
    c0, s0 = np.cos(_TWO_PI * f0 * t), np.sin(_TWO_PI * f0 * t)
    c1, s1 = np.cos(_TWO_PI * f1 * t), np.sin(_TWO_PI * f1 * t)

    x0i, x0q = s * c0, s * (-s0)
    x1i, x1q = s * c1, s * (-s1)

    i0, q0, i1, q1 = state["i0"], state["q0"], state["i1"], state["q1"]
    y_i0 = np.empty_like(x0i)
    y_q0 = np.empty_like(x0q)
    y_i1 = np.empty_like(x1i)
    y_q1 = np.empty_like(x1q)

    for k in range(n):
        i0 += alpha * (x0i[k] - i0)
        q0 += alpha * (x0q[k] - q0)
        i1 += alpha * (x1i[k] - i1)
        q1 += alpha * (x1q[k] - q1)
        y_i0[k], y_q0[k], y_i1[k], y_q1[k] = i0, q0, i1, q1

    # 2) Energías y traza "soft"
    e0 = np.sqrt(y_i0 * y_i0 + y_q0 * y_q0)
    e1 = np.sqrt(y_i1 * y_i1 + y_q1 * y_q1)
    y_soft = e1 - e0

    # 3) Reconstrucción digital (decisión por símbolo)
    bits_hat = np.zeros_like(y_soft)
    num_symbols = n // spb
    for i in range(num_symbols):
        start = i * spb
        end = start + spb
        seg = y_soft[start:end]
        bits_hat[start:end] = 1.0 if np.mean(seg) > 0 else 0.0

    # Último fragmento parcial
    if num_symbols * spb < n:
        seg = y_soft[num_symbols * spb:]
        bits_hat[num_symbols * spb:] = 1.0 if np.mean(seg) > 0 else 0.0

    # 4) Actualizar estado para continuidad
    state["i0"], state["q0"], state["i1"], state["q1"] = float(i0), float(q0), float(i1), float(q1)

    return y_soft, bits_hat

