# core/statistics/metrics.py
import numpy as np
from collections import deque
from typing import Optional, Dict, Any

NCC_DEFAULT_MAXLEN = 12

def _ncc_value(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> Optional[float]:
    """
    Normalized cross-correlation (Pearson r) en [-1, 1] para arrays 1D de igual longitud.
    Devuelve None si no se puede calcular (varianza ~0 o shapes inválidos).
    """
    if x is None or y is None:
        return None
    if x.shape != y.shape:
        return None
    if x.ndim != 1:
        x = x.reshape(-1)
        y = y.reshape(-1)
        if x.shape != y.shape:
            return None

    xm = x - float(np.mean(x))
    ym = y - float(np.mean(y))

    denom = float(np.linalg.norm(xm) * np.linalg.norm(ym))
    if denom < eps:
        return None

    r = float(np.dot(xm, ym) / denom)
    if r > 1.0: r = 1.0
    elif r < -1.0: r = -1.0
    return r

def ncc_percent(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> Optional[float]:
    """
    NCC como porcentaje en [0, 100] usando |r|.
    """
    r = _ncc_value(x, y, eps=eps)
    if r is None:
        return None
    return abs(r) * 100.0

class NCCPairer:
    """
    Empareja (chunk_id, bloque) de 'original' y 'demod' y calcula NCC%.
    - No toca rings.
    - Memoria acotada con deques (maxlen).
    - Solo NCC; sin cálculos extra innecesarios.
    """

    def __init__(self, maxlen: int = NCC_DEFAULT_MAXLEN, eps: float = 1e-12):
        self.maxlen = int(maxlen)
        self.eps = float(eps)
        self._orig_map: Dict[int, np.ndarray] = {}
        self._orig_order: deque[int] = deque()
        self._demod_map: Dict[int, np.ndarray] = {}
        self._demod_order: deque[int] = deque()

    def _evict_if_needed(self, which: str) -> None:
        if which == "orig":
            while len(self._orig_order) > self.maxlen:
                old_id = self._orig_order.popleft()
                self._orig_map.pop(old_id, None)
        elif which == "demod":
            while len(self._demod_order) > self.maxlen:
                old_id = self._demod_order.popleft()
                self._demod_map.pop(old_id, None)

    def push_original(self, chunk_id: int, block: np.ndarray) -> None:
        if chunk_id in self._orig_map:
            try: self._orig_order.remove(chunk_id)
            except ValueError: pass
        self._orig_map[chunk_id] = np.asarray(block, dtype=np.float32).reshape(-1)
        self._orig_order.append(chunk_id)
        self._evict_if_needed("orig")

    def push_demodulated(self, chunk_id: int, block: np.ndarray) -> Optional[Dict[str, Any]]:
        if chunk_id in self._demod_map:
            try: self._demod_order.remove(chunk_id)
            except ValueError: pass
        self._demod_map[chunk_id] = np.asarray(block, dtype=np.float32).reshape(-1)
        self._demod_order.append(chunk_id)
        self._evict_if_needed("demod")

        x = self._orig_map.get(chunk_id)
        y = self._demod_map.get(chunk_id)
        if x is None or y is None:
            return None

        r = _ncc_value(x, y, eps=self.eps)
        if r is None:
            # Limpia esta pareja si no se pudo calcular
            self._orig_map.pop(chunk_id, None)
            self._demod_map.pop(chunk_id, None)
            try: self._orig_order.remove(chunk_id)
            except ValueError: pass
            try: self._demod_order.remove(chunk_id)
            except ValueError: pass
            return None

        result = {"chunk_id": int(chunk_id), "ncc": float(r), "percent": float(abs(r) * 100.0)}

        # Limpia pareja ya usada
        self._orig_map.pop(chunk_id, None)
        self._demod_map.pop(chunk_id, None)
        try: self._orig_order.remove(chunk_id)
        except ValueError: pass
        try: self._demod_order.remove(chunk_id)
        except ValueError: pass

        return result
