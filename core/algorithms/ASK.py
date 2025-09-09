# ask_pcm_end_to_end_improved.py
# ASK end-to-end con mejoras de fidelidad:
# - PCM 8-bit -> bits -> ASK -> demod coherente -> decisión por Integrate&Dump -> bits -> PCM
# - Secuencia piloto 1010... para calibrar umbral y escala
# - LPF FIR (ventana Hamming) y filtro casado rectangular por bit

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# ==========================
# CONFIGURACIÓN (EDITABLE)
# ==========================
INPUT_WAV          = r"assets/ArianaGrande.wav"
OUTPUT_WAV_MOD     = r"assets/ArianaGrande_ASK_mod.wav"
OUTPUT_WAV_DEMOD   = r"assets/ArianaGrande_ASK_demod.wav"

# Audio de trabajo (compromiso calidad/tiempo)
FS_PCM             = 4000        # Hz (sube a 6000–8000 si quieres más banda)
BITS_PER_SAMPLE    = 8           # PCM lineal 8 bits
NORMALIZE_AUDIO    = True

# ASK
SPB                = 12          # samples per bit (>=10 mejora robustez)
AC                 = 0.9         # amplitud portadora
PILOT_BITS         = 400         # longitud de piloto 1010... (>=200 recomendado)

# Portadora
FC                 = None        # None => automático en función de FS_MOD; o fija ej. 12000.0 Hz

# Demod coherente
LPF_ROLL_OFF       = 0.45        # ancho de LPF baseband relativo a 1/SPB (0.35–0.5)
LPF_TAPS           = 129         # longitud del FIR LPF (impar)
# ==========================


# --------- Utilidades ---------
def linear_resample(x, Fs_in, Fs_out):
    if Fs_out == Fs_in:
        return x.astype(np.float32), int(Fs_in)
    dur = len(x)/float(Fs_in)
    t_in = np.arange(len(x), dtype=np.float64)/float(Fs_in)
    n_out = int(round(dur*Fs_out))
    t_out = np.arange(n_out, dtype=np.float64)/float(Fs_out)
    y = np.interp(t_out, t_in, x).astype(np.float32)
    return y, int(Fs_out)

def pcm8u_encode(x):
    x_clip = np.clip(x, -1.0, 1.0)
    return ((x_clip + 1.0) * 0.5 * 255.0).round().astype(np.uint8)

def pcm8u_decode(u):
    return (u.astype(np.float32) / 255.0) * 2.0 - 1.0

def bytes_to_bits(u8):
    return np.unpackbits(u8, bitorder="big").astype(np.uint8)

def bits_to_bytes(b):
    if len(b) % 8 != 0:
        b = b[: len(b)//8 * 8]
    return np.packbits(b.astype(np.uint8), bitorder="big")

def hamming_fir_lowpass(cutoff_norm, taps):
    """
    FIR LP (ventana Hamming) por sinc:
    cutoff_norm en [0, 0.5) relativo a Fs=1 (Nyquist=0.5)
    taps impar recomendado.
    """
    if taps % 2 == 0:
        taps += 1
    M = taps - 1
    n = np.arange(taps) - M/2
    # evitar división por cero en sinc
    h = np.sinc(2*cutoff_norm * n)
    w = 0.54 - 0.46 * np.cos(2*np.pi*np.arange(taps)/M)
    h = h * w
    h = h / np.sum(h)
    return h.astype(np.float32)

def integrate_and_dump(x, spb):
    """
    Integra por ventanas de tamaño spb (filtro casado rectangular).
    Devuelve una muestra por bit (suma/energía).
    """
    nbits = len(x)//spb
    x = x[:nbits*spb]
    X = x.reshape(nbits, spb).sum(axis=1)
    return X.astype(np.float32)

# --------- Lectura y preprocesado ---------
x_in, Fs_in = sf.read(INPUT_WAV, dtype="float32", always_2d=False)
if x_in.ndim == 2:
    x_in = x_in.mean(axis=1).astype(np.float32)
if NORMALIZE_AUDIO:
    peak = np.max(np.abs(x_in)) + 1e-12
    x_in = (x_in / peak).astype(np.float32)

# Remuestrear a FS_PCM
x_pcm, Fs_pcm = linear_resample(x_in, Fs_in, FS_PCM)

# Codificación PCM 8-bit -> bits
u8 = pcm8u_encode(x_pcm)
bits_payload = bytes_to_bits(u8)

# Añadir piloto 1010...
pilot = np.arange(PILOT_BITS, dtype=np.uint8) % 2  # 1010...
bits = np.concatenate([pilot, bits_payload], axis=0)

# Parámetros de transmisión
Rb = BITS_PER_SAMPLE * Fs_pcm          # bits por segundo
FS_MOD = int(Rb * SPB)                 # Fs del modulador
if FC is None:
    # Portadora agradablemente separada (≈ FS_MOD/8)
    fc = max(1000.0, FS_MOD / 8.0)
else:
    fc = float(FC)
if fc >= 0.49*FS_MOD:
    raise ValueError(f"FC={fc} demasiado alta para FS_MOD={FS_MOD} (Nyquist={FS_MOD/2}).")

# --------- Modulación ASK ---------
b_up = np.repeat(bits.astype(np.float32), SPB)
n = np.arange(len(b_up), dtype=np.float32)
carrier = np.cos(2.0*np.pi*fc*n/FS_MOD).astype(np.float32)
s = (b_up * AC) * carrier
# headroom
s_peak = np.max(np.abs(s)) + 1e-12
if s_peak > 0.99:
    s *= (0.99/s_peak)

# Guardar modulada (para escuchar “radio”)
sf.write(OUTPUT_WAV_MOD, s, FS_MOD, subtype="PCM_16")

# --------- Demod coherente ---------
# Mezcla con cos(2π f_c t) -> baseband (contiene banda base alrededor de 0 Hz)
v = 2.0 * s * carrier  # factor 2 para recuperar escala
# LPF FIR para limpiar alta frecuencia (usar normalización relativa a Fs=FS_MOD)
# banda base útil ≈ Rb/2, y el símbolo tiene BW ≈ 1/Tb = Rb
# Normalizamos cutoff respecto a Fs: cutoff_norm = (LPF_ROLL_OFF / SPB)
cutoff_norm = min(0.49, LPF_ROLL_OFF / float(SPB))  # relativo a FS_MOD
h = hamming_fir_lowpass(cutoff_norm, LPF_TAPS)
v_lp = np.convolve(v.astype(np.float32), h, mode="same")

# --------- Integrate & Dump (decisión por bit) ---------
# Saltamos el retardo de grupo del FIR centrado (aprox M/2)
grp_delay = (len(h)-1)//2
v_lp_shift = np.roll(v_lp, -grp_delay)
# Alinear a múltiplos de SPB (descartar cola)
v_lp_shift = v_lp_shift[: (len(v_lp_shift)//SPB)*SPB]
# Sufijo: vector por-bit (suma rectangular)
metric = integrate_and_dump(v_lp_shift, SPB)

# Calibración con piloto: estimar medias de '1' y '0' y umbral intermedio
pilot_len_bits = min(PILOT_BITS, len(metric))
pilot_metric = metric[:pilot_len_bits]
pilot_bits   = pilot[:pilot_len_bits]
mu1 = pilot_metric[pilot_bits == 1].mean() if np.any(pilot_bits==1) else pilot_metric.mean()
mu0 = pilot_metric[pilot_bits == 0].mean() if np.any(pilot_bits==0) else 0.0
thr = 0.5*(mu0 + mu1)

# Decisión final de bits
b_hat = (metric >= thr).astype(np.uint8)

# Quitar piloto
b_hat_payload = b_hat[pilot_len_bits:]
# Asegurar múltiplo de 8
b_hat_payload = b_hat_payload[: (len(b_hat_payload)//8)*8]

# Reconstruir PCM 8-bit
u8_hat = bits_to_bytes(b_hat_payload)
x_pcm_hat = pcm8u_decode(u8_hat)

# Recortar/igualar longitudes a x_pcm
L = min(len(x_pcm_hat), len(x_pcm))
x_pcm_hat = x_pcm_hat[:L]
x_pcm_trim = x_pcm[:L]

# (Opcional) normalizar salida PCM al rango del original remuestreado
# Mantener volumen comparable:
x_pcm_hat *= (np.max(np.abs(x_pcm_trim))+1e-12)

# Remuestrear a Fs original para entrega
x_out, _ = linear_resample(x_pcm_hat, Fs_pcm, Fs_in)
sf.write(OUTPUT_WAV_DEMOD, x_out, Fs_in, subtype="PCM_16")

# --------- Plots breves ---------
ms = 25.0
Nw_pcm = int(min(len(x_pcm),     ms*1e-3*Fs_pcm))
Nw_mod = int(min(len(s),         ms*1e-3*FS_MOD))
t_pcm  = np.arange(Nw_pcm)/Fs_pcm
t_mod  = np.arange(Nw_mod)/FS_MOD

plt.figure(figsize=(12,9))
plt.subplot(3,1,1)
plt.plot(t_pcm, x_pcm[:Nw_pcm])
plt.title(f"Original (resample a {Fs_pcm} Hz) — {ms} ms")
plt.ylabel("Amp")

plt.subplot(3,1,2)
plt.plot(t_mod, s[:Nw_mod])
plt.title(f"ASK Modulada — FS_MOD={FS_MOD} Hz, Rb={BITS_PER_SAMPLE*Fs_pcm:.0f} bps, SPB={SPB}, fc={fc:.0f} Hz, Ac={AC:.2f}")
plt.ylabel("Amp")

plt.subplot(3,1,3)
plt.plot(t_pcm, x_pcm_hat[:Nw_pcm])
plt.title(f"Demod ASK (coherente + I&D) — Umbral≈{thr:.3f}")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amp")
plt.tight_layout()
plt.show()

# --------- Calidad en dominio PCM de trabajo ---------
x0 = x_pcm_trim - np.mean(x_pcm_trim)
y0 = x_pcm_hat   - np.mean(x_pcm_hat)
corr = float(np.corrcoef(x0, y0)[0,1]) if len(x0)>1 else 0.0
print(f"OK ASK: Fs_in={Fs_in} | Fs_pcm={Fs_pcm} | Rb={BITS_PER_SAMPLE*Fs_pcm:.0f} | FS_MOD={FS_MOD} | fc={fc:.0f} | SPB={SPB}")
print(f"Correlación (PCM trabajo): {corr:.3f}")
print(f"Modulado:   {OUTPUT_WAV_MOD}")
print(f"Demodulado: {OUTPUT_WAV_DEMOD}")
