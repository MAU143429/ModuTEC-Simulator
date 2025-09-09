# modu_am_ask_constants.py
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# ==========================
# CONFIGURACIÓN (EDITABLE)
# ==========================
INPUT_WAV         = r"assets/ArianaGrande.wav"
OUTPUT_WAV_MOD    = r"assets/ArianaGrande_modulado.wav"
OUTPUT_WAV_DEMOD  = r"assets/ArianaGrande_demodulado.wav"

FS_OUT      = "keep"        # "keep" o entero (Hz) — usa 'keep' para evitar resample interno
MOD_TYPE    = "AM"          # "AM" (recomendado para audio) o "ASK"

# Parámetros comunes
FC          = "auto"        # "auto" o Hz (p.ej. 20000)
PLOT_MS     = 25.0          # ventana de visualización (ms)

# ---- AM (audio analógico) ----
AC_AM       = "auto"        # "auto" o float (amplitud de portadora)
MU          = 0.8           # índice de modulación (0< MU <=1 para evitar sobre-modulación)
DEMOD_AM    = "envelope"    # "envelope" o "coherent"
AM_SMOOTH_FRAC = 0.15       # fracción de muestras-por-ciclo p/ suavizado de envolvente (0.1–0.3)

# ---- ASK (datos binarios) ----
RB          = 200.0         # bitrate (bits/s) — educativo; NO recupera audio original
AC_ASK      = "auto"        # "auto" o float
BIT_MODE    = "sign"        # cómo sacar bits del audio por ventanas: "sign" o "energy"
ENERGY_TH   = 0.2           # umbral si BIT_MODE = "energy"
ASK_SMOOTH_FRAC = 0.25      # suavizado p/ demod ASK
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


# 1) Leer WAV mono float32
x, Fs_in = sf.read(INPUT_WAV, dtype="float32", always_2d=False)
if x.ndim == 2:
    x = x.mean(axis=1).astype(np.float32)

# 2) Fs de trabajo
if FS_OUT == "keep":
    Fs = Fs_in
else:
    Fs = int(FS_OUT)
    if Fs != Fs_in:
        raise ValueError("Para mantener bajo consumo de RAM, usa FS_OUT='keep' o resamplea fuera del script.")

# 3) Portadora
if FC == "auto":
    fmax = estimate_fmax_fft(x, Fs)
    fc = min(0.25*Fs, max(100.0, 10.0*fmax))  # 10× regla, limitado por Nyquist/4
else:
    fc = float(FC)

n = np.arange(len(x), dtype=np.float32)
carrier = np.cos(2.0*np.pi*fc*n/Fs).astype(np.float32)

# 4) Normalizar audio para AM
x_norm = x / (np.max(np.abs(x)) + 1e-12)

if MOD_TYPE.upper() == "AM":
    # ====== AM (para preservar el audio) ======
    # s = Ac * (1 + MU * m) * cos(2π f_c t), con m en [-1,1]
    if AC_AM == "auto":
        Ac = 0.7 * robust_peak(x_norm)  # headroom ~ -3 dB
        # si x_norm está en [-1,1], robust_peak(x_norm)≈1 → Ac≈0.7
    else:
        Ac = float(AC_AM)
    mu = float(MU)
    mu = max(0.0, min(mu, 1.0))  # evitar sobre-modulación

    s = (Ac * (1.0 + mu * x_norm) * carrier).astype(np.float32)

    # Demodulación
    if DEMOD_AM.lower() == "coherent":
        # Mezcla coherente a baseband + LPF
        v = (2.0 * s * carrier).astype(np.float32)  # contiene DC + m(t)*mu*Ac
        # Suavizado: media móvil ~ LPF
        # longitud en muestras: una fracción de ciclos de la portadora
        samples_per_cycle = max(1, int(Fs / max(1.0, fc)))
        M = max(1, int(samples_per_cycle * AM_SMOOTH_FRAC))
        base = moving_average(v, M)
        # Quitar DC y escalar a ~x_norm
        base = base - np.mean(base)
        # Escala aproximada: dividir por (mu*Ac) (evitando 0)
        scale = (mu*Ac) if (mu*Ac) > 1e-9 else 1.0
        demod = (base / scale).astype(np.float32)
    else:
        # Envolvente: |s| + LPF → ≈ Ac*(1+mu*m)
        env = np.abs(s)
        samples_per_cycle = max(1, int(Fs / max(1.0, fc)))
        M = max(1, int(samples_per_cycle * AM_SMOOTH_FRAC))
        env_lp = moving_average(env, M)
        # Recuperar m: (env/A_c - 1)/mu
        denom = (mu * (Ac + 1e-12))
        demod = ((env_lp / (Ac + 1e-12)) - 1.0) / (mu if denom > 1e-12 else 1.0)
        demod = demod.astype(np.float32)

    # Limitar y guardar
    mod_peak = np.max(np.abs(s)) + 1e-12
    if mod_peak > 0.99:
        s *= (0.99/mod_peak)

    # Ajuste de ganancia en demod para compararlo con x
    demod = demod / (np.max(np.abs(demod)) + 1e-12)
    demod = demod * (np.max(np.abs(x)) + 1e-12)

    # Plots
    Nw = int(min(len(x), PLOT_MS*1e-3*Fs))
    t = n[:Nw] / Fs
    plt.figure(figsize=(12,8))
    plt.subplot(3,1,1)
    plt.plot(t, x[:Nw])
    plt.title(f"Original")
    plt.ylabel("Amp")

    plt.subplot(3,1,2)
    plt.plot(t, s[:Nw])
    plt.title(f"Modulado AM")
    plt.ylabel("Amp")

    plt.subplot(3,1,3)
    plt.plot(t, demod[:Nw])
    plt.title(f"Demodulado AM")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amp")
    plt.tight_layout()
    plt.show()

    # Guardar WAVs
    sf.write(OUTPUT_WAV_MOD, s, Fs, subtype="PCM_16")
    sf.write(OUTPUT_WAV_DEMOD, demod, Fs, subtype="PCM_16")

    # Correlación (medida de similitud)
    L = min(len(x), len(demod))
    x0 = x[:L] - np.mean(x[:L])
    y0 = demod[:L] - np.mean(demod[:L])
    corr = float(np.corrcoef(x0, y0)[0,1])
    print(f"[AM] fc={fc:.1f} Hz, Ac={Ac:.4f}, mu={mu:.3f} | correlación original vs demod: {corr:.3f}")

else:
    # ====== ASK (binario, educativo: NO preserva la forma de onda) ======
    # s = b_up * Ac * cos(2π f_c t)
    if AC_ASK == "auto":
        Ac = 0.7 * robust_peak(x)  # headroom
    else:
        Ac = float(AC_ASK)

    Rb = float(RB)
    Tb = 1.0 / Rb
    spb = int(max(1, round(Tb * Fs)))
    nbits = max(1, int(len(x) // spb))

    b = np.zeros(nbits, dtype=np.uint8)
    for k in range(nbits):
        i0 = k*spb
        i1 = min(len(x), i0+spb)
        w = x[i0:i1]
        if BIT_MODE == "energy":
            rms = np.sqrt(np.mean(w*w) + 1e-12)
            b[k] = 1 if rms >= ENERGY_TH else 0
        else:
            b[k] = 1 if np.mean(w) >= 0.0 else 0

    b_up = np.repeat(b.astype(np.float32), spb)
    if len(b_up) < len(x):
        b_up = np.pad(b_up, (0, len(x)-len(b_up)))
    elif len(b_up) > len(x):
        b_up = b_up[:len(x)]

    s = (b_up * Ac) * carrier
    mod_peak = np.max(np.abs(s)) + 1e-12
    if mod_peak > 0.99:
        s *= (0.99/mod_peak)

    # Demod ASK (envelope o coherente a elección)
    # Para simplicidad, envelope:
    env = np.abs(s)
    M = max(1, int(spb * ASK_SMOOTH_FRAC))
    demod_env = moving_average(env, M).astype(np.float32)

    # Normalizar demod para escuchar cómo queda (será "granulado", no igual al original)
    demod_out = demod_env / (np.max(np.abs(demod_env)) + 1e-12)
    demod_out *= (np.max(np.abs(x)) + 1e-12)

    # Plots
    Nw = int(min(len(x), PLOT_MS*1e-3*Fs))
    t = n[:Nw] / Fs
    plt.figure(figsize=(12,8))
    plt.subplot(3,1,1)
    plt.plot(t, x[:Nw]); plt.title(f"Original (primeros {PLOT_MS} ms) — Fs={Fs} Hz"); plt.ylabel("Amp")
    plt.subplot(3,1,2)
    plt.plot(t, s[:Nw]); plt.title(f"ASK (primeros {PLOT_MS} ms) — Rb={Rb} bps, fc={fc:.1f} Hz, Ac={Ac:.3f}"); plt.ylabel("Amp")
    plt.subplot(3,1,3)
    plt.plot(t, demod_out[:Nw]); plt.title("Demodulado ASK (envolvente, NO igual al original)"); plt.xlabel("Tiempo [s]"); plt.ylabel("Amp")
    plt.tight_layout(); plt.show()

    # Guardar WAVs
    sf.write(OUTPUT_WAV_MOD, s, Fs, subtype="PCM_16")
    sf.write(OUTPUT_WAV_DEMOD, demod_out, Fs, subtype="PCM_16")

    # Correlación (no será alta porque es binario)
    L = min(len(x), len(demod_out))
    x0 = x[:L] - np.mean(x[:L])
    y0 = demod_out[:L] - np.mean(demod_out[:L])
    corr = float(np.corrcoef(x0, y0)[0,1])
    print(f"[ASK] Rb={Rb:.1f} bps, fc={fc:.1f} Hz, Ac={Ac:.4f} | correlación original vs demod: {corr:.3f} (espera baja)")
