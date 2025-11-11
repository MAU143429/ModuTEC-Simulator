import numpy as np
import math

# ============================================================
# ASK (OOK) por bloques con digitalización adaptativa
# ------------------------------------------------------------
# MOD:
#   - Por-bloque calcula un umbral tau_blk = percentil(|x|, p)
#   - Promedia |x| por símbolo y decide 0/1 con histéresis + debounce
#   - Genera s_mod = Ac * cos(2πfc t + phase) * NRZ
#
# DEMOD:
#   - Envolvente LPF de |s|
#   - Promedio por símbolo
#   - Decisión con umbral adaptativo tau = 0.5*(mu0+mu1) y mu0/mu1 (EWMA)
#
# Notas:
#   - No se normaliza el PCM globalmente; solo se analiza por bloque
#   - Reloj de símbolo continuo (spb constante)
# ============================================================

def ask_prepare_state(first_chunk: np.ndarray, Fs: float, fc: float, Ac: float, bitrate: float):
    st = {}

    st["Fs"] = float(Fs)
    st["ask_fc"] = float(fc)
    st["ask_Ac"] = float(Ac)
    st["ask_bitrate"] = float(bitrate)

    # --- reloj de símbolo ---
    spb = int(round(Fs / float(bitrate)))
    st["spb"] = max(2, spb)

    # --- parámetros de digitalización MOD ---
    st["thr_percentile"] = 70.0   # percentil aplicado sobre |x| de CADA bloque
    st["hyst_frac"]      = 0.10   # banda de histéresis como fracción de tau_blk
    st["debounce_syms"]  = 1      # símbolos mínimos entre toggles

    # acumuladores por símbolo (MOD)
    st["mod_sym_sum"] = 0.0       # acum de |x|
    st["mod_sym_cnt"] = 0
    st["mod_bit"]     = 0
    st["since_toggle_mod"] = 1e9
    st["last_tau_blk"] = 0.0      # sólo para debug si lo quieres mostrar

    # --- DEMOD: filtro de envolvente 1 polo ---
    # fc_env ≈ 0.35*bitrate (cap a 5% de Fs)
    fc_env = min(0.35 * bitrate, 0.05 * Fs)
    alpha = 1.0 - math.exp(-2.0 * math.pi * fc_env / Fs)
    st["env_alpha"] = float(alpha)
    st["env_lp"] = 0.0

    # niveles adaptativos de la envolvente (EWMA)
    st["mu0"] = 0.10 * Ac
    st["mu1"] = 0.60 * Ac
    st["mu_rho"] = 0.10

    # acumuladores por símbolo (DEMOD)
    st["dem_sym_sum"] = 0.0    # acum de envolvente
    st["dem_sym_cnt"] = 0
    st["dem_bit"]     = 0

    # --- continuidad de portadora ---
    st["phase"] = 0.0
    return st


# ------------------------------------------------------------
# Utilidad: envolvente LPF de |x|
# ------------------------------------------------------------
def _env_step_block(x: np.ndarray, env_lp: float, alpha: float):
    env = np.empty_like(x, dtype=np.float32)
    lp = float(env_lp)
    a  = float(alpha)
    xf = x.astype(np.float32, copy=False)
    for i, v in enumerate(xf):
        if v < 0: v = -v
        lp = (1.0 - a) * lp + a * v
        env[i] = lp
    return env, lp


# ------------------------------------------------------------
# Digitalización MOD: decisión por símbolo usando |x|
# ------------------------------------------------------------
def _nrz_from_abs_symbol_locked(x: np.ndarray, st: dict, tau_blk: float):
    """
    x: bloque PCM (float32). Se usa |x| para robustez frente a signo.
    tau_blk: umbral del BLOQUE (percentil de |x|).
    Regla por símbolo:
      m_sym = mean(|x_sym|)
      si m_sym >= tau_blk + band  -> bit=1
      si m_sym <= tau_blk - band  -> bit=0
      si dentro de banda muerta   -> mantener bit anterior
    con histéresis (band = hyst_frac * tau_blk) y debounce en símbolos.
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
            # si m entra en (down, up) mantiene bit

            sym_sum = 0.0
            sym_cnt = 0

    st["mod_sym_sum"] = float(sym_sum)
    st["mod_sym_cnt"] = int(sym_cnt)
    st["mod_bit"]     = int(bit)
    st["since_toggle_mod"] = int(since)
    return bits, st


# ========================= MOD ==============================
def ask_modulate_block(x_chunk: np.ndarray, state: dict):
    """
    1) Calcula tau_blk = percentil(|x|, thr_percentile)
    2) Digitaliza por símbolo (|x|) con histéresis y debounce
    3) s_mod = Ac * cos(2πfc t + phase) * NRZ
    """
    Fs = float(state["Fs"])
    fc = float(state["ask_fc"])
    Ac = float(state["ask_Ac"])

    # --- 1) Umbral por bloque (robusto)
    p = float(state.get("thr_percentile", 70.0))
    absx = np.abs(x_chunk.astype(np.float32, copy=False))
    tau_blk = float(np.percentile(absx, p))
    tau_blk = max(tau_blk, 1e-9)
    state["last_tau_blk"] = tau_blk  # (opcional debug en UI)

    # --- 2) NRZ por símbolo (sobre |x|)
    bits_nrz, state = _nrz_from_abs_symbol_locked(x_chunk, state, tau_blk)

    # --- 3) Portadora continua
    n = np.arange(len(bits_nrz), dtype=np.float32)
    wc = (2.0 * np.pi * fc) / Fs
    phase0 = float(state.get("phase", 0.0))
    phase = phase0 + wc * n
    s_mod = (Ac * np.cos(phase) * bits_nrz.astype(np.float32)).astype(np.float32)
    state["phase"] = float((phase0 + wc * len(n)) % (2.0 * np.pi))

    stats = {
        "spb": int(state["spb"]),
        "tau_blk": float(tau_blk),
    }
    return s_mod, bits_nrz, state, stats


# ======================== DEMOD =============================
def ask_demodulate_block(s_chunk: np.ndarray, state: dict):
    """
    Demod ASK robusta + salida ya en escala de visualización:
    - LPF de |s| (envolvente)
    - Integrate-and-dump por símbolo
    - Niveles adaptativos mu0/mu1 (EWMA + leak)
    - Histeresis dual (relativa y absoluta) + debounce
    - y_plot: traza mapeada a [-Aplot, +Aplot] para que coincida con la "Señal Original"
    """
    Fs      = float(state["Fs"])
    Ac      = float(state["ask_Ac"])
    spb     = int(state["spb"])
    alpha   = float(state["env_alpha"])
    rho     = float(state.get("mu_rho", 0.10))        # velocidad EWMA
    deb_sy  = int(state.get("debounce_syms", 1))      # debounce en símbolos
    Aplot   = float(state.get("view_amp", 0.1))       # amplitud de plot deseada

    # Knobs de conmutación (ajústalos si quieres más rapidez/estabilidad)
    rel_band_k   = float(state.get("rel_band_k", 0.16))  # 0.12..0.20 (menor = más sensible)
    abs_band_k   = float(state.get("abs_band_k", 0.03))  # 0.03..0.06  (mayor = menos chatter)
    leak_per_blk = float(state.get("leak_per_blk", 0.03))
    gap_floor    = 0.04 * Ac

    # 1) Envolvente LPF
    env = np.empty_like(s_chunk, dtype=np.float32)
    lp  = float(state.get("env_lp", 0.0))
    a   = float(alpha)
    s_f = s_chunk.astype(np.float32, copy=False)
    for i, v in enumerate(s_f):
        av = v if v >= 0 else -v
        lp = (1.0 - a) * lp + a * av
        env[i] = lp
    state["env_lp"] = float(lp)

    # 2) Estadística de bloque para umbral absoluto
    p20  = float(np.percentile(env, 20))
    p80  = float(np.percentile(env, 80))
    nf   = max(1e-9, p20)                        # ruido/piso
    top  = max(p80, 0.5 * Ac, nf + gap_floor)    # alto robusto

    # 3) Niveles adaptativos (con leak hacia stats frescas)
    mu0  = float(state.get("mu0", 0.10 * Ac))
    mu1  = float(state.get("mu1", 0.60 * Ac))
    mu0  = (1.0 - leak_per_blk) * mu0 + leak_per_blk * nf
    mu1  = (1.0 - leak_per_blk) * mu1 + leak_per_blk * top

    # 4) Integrate-and-dump por símbolo + Schmitt con debounce
    sym_sum = float(state.get("dem_sym_sum", 0.0))
    sym_cnt = int(state.get("dem_sym_cnt", 0))
    bit     = int(state.get("dem_bit", 0))
    since   = int(state.get("since_toggle", 1000))
    bits_hat = np.empty_like(env, dtype=np.uint8)

    gap   = max(gap_floor, (mu1 - mu0))
    rel_band = rel_band_k * gap
    abs_band = abs_band_k * max(nf, 1e-9)

    # Además generamos un "soft-NRZ" normalizado para la traza
    # n_env = clip((env - mu0)/gap, 0..1)
    n_env = (env - mu0) / (gap + 1e-12)
    n_env = np.clip(n_env, 0.0, 1.0)

    for i, e in enumerate(env):
        sym_sum += float(e)
        sym_cnt += 1
        bits_hat[i] = bit
        since += 1

        if sym_cnt >= spb:
            m   = sym_sum / float(sym_cnt)      # media por símbolo
            tau = 0.5 * (mu0 + mu1)
            band = max(rel_band, abs_band)

            # Schmitt + debounce
            if bit == 0 and (m >= tau + band) and since >= deb_sy:
                bit = 1; since = 0
            elif bit == 1 and (m <= tau - band) and since >= deb_sy:
                bit = 0; since = 0

            # EWMA acotado
            if bit == 0:
                target = min(m, tau)
                mu0 = (1.0 - rho) * mu0 + rho * target
            else:
                target = max(m, tau)
                mu1 = (1.0 - rho) * mu1 + rho * target

            sym_sum = 0.0
            sym_cnt = 0
            gap     = max(gap_floor, (mu1 - mu0))
            rel_band = rel_band_k * gap
            abs_band = abs_band_k * max(nf, 1e-9)

    # Persistencia
    state["mu0"] = float(mu0)
    state["mu1"] = float(mu1)
    state["dem_sym_sum"] = float(sym_sum)
    state["dem_sym_cnt"] = int(sym_cnt)
    state["dem_bit"] = int(bit)
    state["since_toggle"] = int(since)

    # 5) Salida para el plot: MISMA ESCALA QUE LA ORIGINAL
    # Mapear n_env ∈ [0,1] → [-Aplot, +Aplot] (Aplot≈0.1)
    y_plot = (n_env * (2.0 * Aplot) - Aplot).astype(np.float32)

    return y_plot, bits_hat

