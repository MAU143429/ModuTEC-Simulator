import numpy as np

_TWO_PI = 2.0 * np.pi

# ---------------------------
# Utilidades por-bloque (AM)
# ---------------------------

def estimate_fmax_block(x: np.ndarray, Fs: float, nfft: int = 2048, energy_pct: float = 0.99) -> float:
    """
    Estima una fmax del contenido baseband del bloque usando un percentil
    de energía del espectro. Ligero y estable para ventanas pequeñas.
    """
    if x is None or len(x) == 0 or Fs <= 0:
        return 1000.0
    L = len(x)
    if L < nfft:
        seg = np.zeros(nfft, dtype=np.float32)
        seg[:L] = x.astype(np.float32, copy=False)
        seg *= np.hanning(nfft)
    else:
        seg = x[:nfft].astype(np.float32, copy=False) * np.hanning(nfft)
    X = np.fft.rfft(seg)
    P = (np.abs(X) ** 2) / nfft
    P = np.maximum(P, 0.0)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / float(Fs))
    cum = np.cumsum(P)
    tot = cum[-1] if cum.size and cum[-1] > 0 else 1.0
    idx = np.searchsorted(cum, energy_pct * tot)
    return float(freqs[min(idx, len(freqs) - 1)])

def compute_blk_stats(x: np.ndarray, Fs: float) -> dict:
    """
    Calcula estadísticas por bloque para AM adaptativa.
    Devuelve: {'blk_mean','blk_rms','blk_peak','blk_fmax'}
    """
    if x is None or x.size == 0:
        return {"blk_mean": 0.0, "blk_rms": 0.0, "blk_peak": 0.0, "blk_fmax": 1000.0}
    x64 = x.astype(np.float64, copy=False)
    mean = float(np.mean(x64))
    xdc  = x64 - mean
    rms  = float(np.sqrt(np.mean(xdc * xdc)))
    peak = float(np.max(np.abs(xdc)))
    fmax = estimate_fmax_block(x.astype(np.float32, copy=False), Fs)
    return {"blk_mean": mean, "blk_rms": rms, "blk_peak": peak, "blk_fmax": fmax}

# ---------------------------
# Núcleo AM
# ---------------------------

def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a / (b if abs(b) > eps else eps))

def am_modulate_block(x: np.ndarray, Fs: float, state: dict) -> tuple[np.ndarray, dict]:
    """
    AM coherente ADAPTATIVA por bloque:
      s[n] = Ac * (1 + mu * x_norm[n]) * cos(2π fc n/Fs + φ0)
    Donde x_norm se centra con blk_mean y escala con blk_peak del propio bloque.
    Espera en state: fc, mu, Ac, phase, blk_mean, blk_peak
    """
    fc  = float(state["fc"])
    mu  = float(state["mu"])
    Ac  = float(state["Ac"])
    ph0 = float(state["phase"])

    blk_mean = float(state.get("blk_mean", 0.0))
    blk_peak = float(state.get("blk_peak", 1.0))

    x_f = x.astype(np.float32, copy=False)
    x_c = (x_f - np.float32(blk_mean)).astype(np.float32, copy=False)
    x_scale = np.float32(blk_peak if blk_peak > 1e-12 else 1.0)
    x_n = (x_c / x_scale).astype(np.float32, copy=False)

    L  = len(x_n)
    n  = np.arange(L, dtype=np.float32)
    omg = _TWO_PI * fc / float(Fs)
    ph  = ph0 + omg * n
    car = np.cos(ph, dtype=np.float32)

    s = (Ac * (1.0 + mu * x_n) * car).astype(np.float32, copy=False)

    ph1 = ph0 + omg * L
    ph1 = float(np.fmod(ph1, _TWO_PI))

    st = dict(state)
    st["phase"] = ph1
    return s, st

def am_demodulate_block(s: np.ndarray, Fs: float, state: dict) -> tuple[np.ndarray, dict]:
    """
    Demodulación coherente adaptativa:
      - Mezcla con portadora local
      - LPF 1-polo con fc definido por blk_fmax del bloque
      - Ajuste físico con (Ac, mu) y recentrado en DC
    Espera en state: fc, mu, Ac, phase, blk_fmax, lp_ym1
    """
    fc   = float(state.get("fc", 1000.0))
    mu   = float(state.get("mu", 0.5))
    Ac   = float(state.get("Ac", 1.0))
    fmax = float(state.get("blk_fmax", 1000.0))
    ph0  = float(state.get("phase", 0.0))
    y0   = float(state.get("lp_ym1", 0.0))

    L   = len(s)
    n   = np.arange(L, dtype=np.float32)
    omg = _TWO_PI * fc / float(Fs)
    ph  = ph0 + omg * n

    ref   = np.cos(ph, dtype=np.float32)
    mixed = ( 2.0 * s.astype(np.float32, copy=False) * ref).astype(np.float32, copy=False)

    # corte LPF: proporcional a fmax con leve margen/suelo
    #cut = min(0.15 * Fs, max(1.1 * fmax, fmax + 200.0))
    # objetivo de corte (baseband + margen razonable)
    cut_target = max(1.3 * fmax, fmax + 200.0)
    cut_target = min(cut_target, 0.10 * Fs)   # techo 10% Fs
    cut_target = max(cut_target, 40.0)        # piso 40 Hz

    # FIR lineal-fase (retardo constante ya compensado en fir_lpf_block)
    y_lp, st = fir_lpf_block(mixed, Fs, cut_target, dict(state), taps=129)
    # si quieres guardar estado FIR en 'st' para bloques siguientes:
    state.update({k: v for k, v in st.items() if k.startswith("fir_")})
    
    y = (y_lp / (Ac + 1e-12) - 1.0) / (mu + 1e-12)
    blk_mean = float(state.get("blk_mean", 0.0))
    blk_peak = float(state.get("blk_peak", 1.0))
    y_rec = y * blk_peak + blk_mean

    return y_rec.astype(np.float32), st


def one_pole_lpf_block(x: np.ndarray, Fs: float, fc_hz: float, y_prev: float) -> tuple[np.ndarray, float]:
    """ LPF 1-polo con estado persistente entre bloques. """
    fc    = max(1.0, float(fc_hz))
    alpha = 1.0 - np.exp(-2.0 * np.pi * fc / float(Fs))
    y     = np.empty_like(x, dtype=np.float32)
    y_last = float(y_prev)
    x_f   = x.astype(np.float32, copy=False)
    for i in range(len(x_f)):
        y_last = y_last + alpha * (float(x_f[i]) - y_last)
        y[i]   = y_last
    return y, y_last

# --------------------------------------------
# Conveniencia: paso AM completo por bloque
# --------------------------------------------
def am_process_block(x: np.ndarray, Fs: float, state: dict) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Hace TODO el ciclo AM para un bloque:
      - Calcula estadísticas por bloque
      - Modula
      - Demodula
      - Devuelve s_mod, s_demod, state_actualizado, stats
    Requiere en state: fc, mu, Ac, phase (y opcional lp_ym1)
    """
    stats = compute_blk_stats(x, Fs)
    
    st_in = dict(state)
    

    st_in.update({
        "blk_mean": stats["blk_mean"],
        "blk_peak": stats["blk_peak"],
        "blk_fmax": stats["blk_fmax"],
    })

    s_mod, st_mid   = am_modulate_block(x, Fs, st_in)
    s_dem, st_final = am_demodulate_block(s_mod, Fs, st_mid)
    return s_mod, s_dem, st_final, stats


# ---------- FIR LPF lineal-fase con ventana Hamming ----------
def _fir_lpf_design(fc_hz: float, Fs: float, taps: int = 129) -> np.ndarray:
    fc = max(10.0, min(fc_hz, 0.12*Fs))
    wc = 2.0*np.pi*fc/float(Fs)
    M  = int(taps)
    n  = np.arange(M, dtype=np.float64)
    m  = n - (M-1)/2.0
    h  = np.where(m == 0.0, wc/np.pi, np.sin(wc*m)/(np.pi*m))
    w  = 0.54 - 0.46*np.cos(2.0*np.pi*n/(M-1))
    h  = (h*w).astype(np.float32, copy=False)
    h /= (np.sum(h) + 1e-12)   # DC = 1
    return h

def fir_lpf_block(x: np.ndarray, Fs: float, fc_hz: float, state: dict,
                  taps: int = 129) -> tuple[np.ndarray, dict]:
    # reusa coeficientes si fc no cambió
    h  = state.get("fir_h", None)
    fch= state.get("fir_fc", None)
    if (h is None) or (fch is None) or abs(fch - fc_hz) > 1e-3:
        h = _fir_lpf_design(fc_hz, Fs, taps=taps)
        state["fir_h"]  = h
        state["fir_fc"] = float(fc_hz)

    # overlap con memoria simple
    z = state.get("fir_zi", np.zeros(len(h)-1, dtype=np.float32))
    y_full = np.convolve(x.astype(np.float32, copy=False), h, mode="full")
    y_full[:len(z)] += z
    z = y_full[-(len(h)-1):].copy()
    y = y_full[:len(x)].astype(np.float32, copy=False)
    state["fir_zi"] = z

    # Compensación exacta del retardo constante D=(M-1)/2
    D = (len(h)-1)//2
    if D > 0:
        y = np.roll(y, -D)
        if D < len(y):
            y[-D:] = y[-D-1]
    return y, state
