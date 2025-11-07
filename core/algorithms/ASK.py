# ASK.py — OOK (On-Off Keying) con mapeo audio→bits y continuidad por bloques
import numpy as np

_TWO_PI = 2.0*np.pi

# ---------- Utilidades internas ----------

def _robust_peak(x: np.ndarray) -> float:
    return float(np.percentile(np.abs(x), 99.0) + 1e-12)

def _alpha_from_fc(fc_hz: float, Fs: float) -> float:
    """Alpha 1-polo desde frecuencia de corte (en Hz)."""
    fc_hz = max(1.0, float(fc_hz))
    return 1.0 - np.exp(-2.0*np.pi*fc_hz / Fs)

# ---------- Estado y preparación ----------

def ask_prepare_state(first_chunk: np.ndarray,
                      Fs: float,
                      fc: float,
                      Ac: float,
                      bitrate: float):
    """
    Prepara parámetros fijos y estado persistente entre bloques para OOK.
    Entradas del usuario: Fs, fc, Ac, bitrate (todos requeridos).
    - spb = Fs/bitrate (redondeado, mínimo 2)
    - xscale: normalización fija (pico robusto del primer bloque)
    - Histéresis adaptativa basada en percentiles del primer bloque
    - LPF de envolvente con fc ~ 0.25*bitrate (suaviza bien símbolos)
    """
    x = first_chunk.astype(np.float64, copy=False)
    xscale = _robust_peak(x)
    spb = max(2, int(round(Fs / max(1.0, float(bitrate)))))

    # Umbrales adaptativos sobre el primer bloque normalizado
    xn = x / xscale
    p95 = float(np.percentile(np.abs(xn), 95.0))
    # histéresis: low < high
    thr_high = 0.35 * p95
    thr_low  = 0.15 * p95

    # LPF para demod (envolvente): un poco más rápido para seguir símbolos
    env_fc = max(1.0, 0.5 * bitrate)           # antes 0.25*bitrate
    env_alpha = _alpha_from_fc(env_fc, Fs)

    state = {
        # Parámetros fijos
        "Fs": float(Fs),
        "fc": float(fc),
        "Ac": float(Ac),
        "bitrate": float(bitrate),
        "spb": int(spb),
        "xscale": float(xscale),

        # Modulación: histéresis + reloj de símbolo
        "mod_thr_low": float(thr_low),
        "mod_thr_high": float(thr_high),
        "mod_state": 0.0,               # estado Schmitt actual (0/1)
        "mod_one_count": 0,             # contador de unos en la ventana símbolo
        "mod_sample_in_sym": 0,         # muestras acumuladas dentro del símbolo
        "mod_curr_bit": 0.0,            # bit vigente (0/1) que se “tiene” hasta el próximo corte
        "phase": 0.0,                   # fase acumulada de la portadora

        # Demodulación: energía + LPF + umbral adaptativo con histéresis
        "env_prev": 0.0,                # memoria del LPF sobre energía
        "env_alpha": float(env_alpha),  # alpha LPF
        "env_mean": 0.0,                # media/ruido adaptativa de la envolvente
        "thr_k": 0.7,                   # umbral = thr_k * env_mean (ajustable 0.7–0.95)
        "hyst": 0.17,                   # ±20% de histéresis
        "demod_state": 0.0,             # estado Schmitt 0/1 sobre envolvente
        "demod_one_count": 0,
        "demod_sample_in_sym": 0,
        "demod_curr_bit": 0.0,
    }
    return state

# ---------- Binarización con histéresis (Schmitt) ----------

def _schmitt_sample(v: float, state_val: float, low: float, high: float) -> float:
    """
    Un paso de histéresis. v debe venir normalizado (≈ [-1,1]).
    state_val es 0.0 o 1.0 actual.
    """
    if state_val <= 0.5:
        # En 0, solo sube si supera el high
        return 1.0 if v >= high else 0.0
    else:
        # En 1, solo baja si cae por debajo de low
        return 0.0 if v <= low else 1.0

# ---------- Modulación ----------

def ask_modulate_block(x_chunk: np.ndarray, state: dict):
    """
    Modulación OOK de un bloque desde PCM:
      1) Normaliza por xscale (fijo del primer bloque)
      2) Binariza por histéresis (Schmitt)
      3) “Ritma” a bitrate por ventanas de spb (majority) manteniendo continuidad entre bloques
      4) s[n] = (bit_actual ? Ac : 0) * cos(2π fc t + phase)

    Devuelve: s (bloque modulado), bits_nrz_por_muestra (0/1 por muestra)
    """
    Fs, fc, Ac, spb = state["Fs"], state["fc"], state["Ac"], state["spb"]
    xscale = state["xscale"]
    thr_low, thr_high = state["mod_thr_low"], state["mod_thr_high"]

    mod_state = state["mod_state"]
    ones = state["mod_one_count"]
    k = state["mod_sample_in_sym"]
    curr_bit = state["mod_curr_bit"]
    phase = state["phase"]

    x = x_chunk.astype(np.float64, copy=False) / xscale
    n = len(x)

    # Salidas
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
            # Decisión de símbolo por mayoría
            curr_bit = 1.0 if (ones >= (spb - ones)) else 0.0
            ones = 0
            k = 0
            # El siguiente sample ya reflejará curr_bit actualizado

    # Portadora y modulación
    t = np.arange(n) / Fs
    carrier = np.cos(_TWO_PI*fc*t + phase)
    s = (bits_nrz * Ac) * carrier

    # Avance de fase continuo
    phase = (phase + _TWO_PI*fc*(n/Fs)) % (2.0*np.pi)

    # Guardar estado
    state["mod_state"] = float(mod_state)
    state["mod_one_count"] = int(ones)
    state["mod_sample_in_sym"] = int(k)
    state["mod_curr_bit"] = float(curr_bit)
    state["phase"] = float(phase)

    return s, bits_nrz

# ---------- Demodulación ----------

def ask_demodulate_block(s_chunk: np.ndarray, state: dict):
    """
    Demod OOK mejorada:
      - Energía: e = s^2
      - LPF 1 polo sobre energía y_env_e -> y_env = sqrt(y_env_e)
      - Umbral adaptativo: thr = max(0.1*Ac, thr_k * mean_env)
      - Schmitt con histéresis (low/high) sobre y_env por muestra
      - Decisión por símbolo (mayoría cada spb)
    Devuelve:
      y_step: escalón 0..Ac por muestra (parecido a la original NRZ)
      bits_hat_nrz: 0/1 por muestra
    """
    Fs, spb = state["Fs"], state["spb"]
    alpha = state["env_alpha"]
    Ac = state["Ac"]
    thr_k = state["thr_k"]
    hyst = state["hyst"]

    s = s_chunk.astype(np.float64, copy=False)
    n = len(s)

    # 1) Energía y LPF
    e = s * s
    y_env_e = np.empty_like(e)
    prev = state["env_prev"]
    for i in range(n):
        prev = prev + alpha*(e[i] - prev)
        y_env_e[i] = prev
    state["env_prev"] = float(prev)

    # 2) Envolvente lineal (amplitud)
    y_env = np.sqrt(y_env_e + 1e-18)

    # 3) Umbral adaptativo con media lenta de la envolvente
    #    (alpha_mean más lento que alpha para estabilidad)
    alpha_mean = max(0.0025, 0.25*alpha)
    mean_env = state["env_mean"]
    for i in range(n):
        mean_env = mean_env + alpha_mean*(y_env[i] - mean_env)
    state["env_mean"] = float(mean_env)
    thr = max(0.1*Ac, thr_k * state["env_mean"])

    low = (1.0 - hyst) * thr
    high = (1.0 + hyst) * thr

    # 4) Schmitt por muestra con deglitch (run-length mínimo)
    st_raw = state["demod_state"]      # estado crudo de Schmitt (0/1)
    acc_state = state.get("acc_state", st_raw)  # estado aceptado (con deglitch)
    run_len = int(state.get("run_len", 0))      # longitud de la racha actual (en muestras)
    min_run = max(2, int(0.5 * spb))            # exigir 60% de un símbolo para aceptar cambio

    bits_hat_nrz = np.empty(n, dtype=np.float64)
    y_step = np.empty(n, dtype=np.float64)

    for i in range(n):
        # Schmitt instantáneo (crudo)
        if st_raw <= 0.5:
            st_raw = 1.0 if y_env[i] >= high else 0.0
        else:
            st_raw = 0.0 if y_env[i] <= low else 1.0

        # Deglitch por run-length: solo acepto cambio si la nueva racha es suficientemente larga
        if st_raw == acc_state:
            run_len += 1
        else:
            # posible cambio
            if run_len >= min_run:
                acc_state = st_raw
                run_len = 1
            else:
                # ignoro el cambio breve (glitch), mantengo acc_state
                run_len += 1

        bits_hat_nrz[i] = acc_state
        y_step[i] = Ac * acc_state

    # Guardar estado
    state["demod_state"] = float(st_raw)
    state["acc_state"] = float(acc_state)
    state["run_len"] = int(run_len)

    return y_step, bits_hat_nrz


