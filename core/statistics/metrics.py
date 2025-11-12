import numpy as np
from collections import deque
from typing import Optional, Dict, Any

# Default maximum length for internal deques used in NCCPairer
NCC_DEFAULT_MAXLEN = 12


# Compute the Normalized Cross-Correlation (Pearson correlation coefficient) between two 1D arrays x and y.
# Returns None if the computation is not possible (e.g., zero variance or mismatched array shapes).
def ncc_percent(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> Optional[float]:
    
    # Validate inputs
    if x is None or y is None:
        return None
    if x.shape != y.shape:
        return None

    # If the arrays are not 1D, flatten them
    if x.ndim != 1:
        x = x.reshape(-1)
        y = y.reshape(-1)
        if x.shape != y.shape:
            return None

    # Subtract the mean from each signal
    xm = x - float(np.mean(x))
    ym = y - float(np.mean(y))

    # Calculate the denominator as the product of L2 norms
    denom = float(np.linalg.norm(xm) * np.linalg.norm(ym))
    if denom < eps:
        return None

    # Pearson correlation formula
    r = float(np.dot(xm, ym) / denom)

    # Clamp result to [-1, 1].
    if r > 1.0:
        r = 1.0
    elif r < -1.0:
        r = -1.0
        
    return abs(r) * 100.0


class NCCPairer:
    """
    This class pairs 'original' and 'demodulated' signal blocks by chunk_id
    and computes the NCC percentage between them.

    Key design characteristics:
    - Uses bounded deques to limit memory growth.
    - Keeps separate maps for original and demodulated chunks.
    - Once a valid pair (same chunk_id) is processed, it is deleted from memory.
    - Focused purely on NCC; minimal additional computation.
    """

    def __init__(self, maxlen: int = NCC_DEFAULT_MAXLEN, eps: float = 1e-12):
        # Maximum number of blocks stored in memory
        self.maxlen = int(maxlen)
        self.eps = float(eps)

        # Internal dictionaries map chunk_id -> signal arrays
        self._orig_map: Dict[int, np.ndarray] = {}
        self._demod_map: Dict[int, np.ndarray] = {}

        # Deques to preserve insertion order and control size
        self._orig_order: deque[int] = deque()
        self._demod_order: deque[int] = deque()

    def _evict_if_needed(self, which: str) -> None:
        """
        Remove oldest entries if deque exceeds maxlen.
        This prevents unbounded growth if new chunks keep arriving.
        """
        if which == "orig":
            while len(self._orig_order) > self.maxlen:
                old_id = self._orig_order.popleft()
                self._orig_map.pop(old_id, None)
        elif which == "demod":
            while len(self._demod_order) > self.maxlen:
                old_id = self._demod_order.popleft()
                self._demod_map.pop(old_id, None)

    def push_original(self, chunk_id: int, block: np.ndarray) -> None:
        """
        Store a new 'original' signal block.

        - Ensures the chunk_id is unique (removes previous if necessary).
        - Converts to float32 and flattens.
        - Adds to the ordered deque and evicts old items if needed.
        """
        if chunk_id in self._orig_map:
            # If this chunk already exists, remove it from the deque
            try:
                self._orig_order.remove(chunk_id)
            except ValueError:
                pass

        # Store and normalize the block
        self._orig_map[chunk_id] = np.asarray(block, dtype=np.float32).reshape(-1)
        self._orig_order.append(chunk_id)
        self._evict_if_needed("orig")

    def push_demodulated(self, chunk_id: int, block: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Store a new 'demodulated' signal block and compute the NCC
        if its matching 'original' block exists.

        Parameters
        ----------
        chunk_id : int
            Identifier for the signal block.
        block : np.ndarray
            Demodulated signal data.

        Returns
        -------
        result : Optional[Dict[str, Any]]
            Dictionary containing {chunk_id, ncc, percent}, or None if
            the pair is incomplete or invalid.
        """
        # Remove duplicates if the same chunk_id was previously stored
        if chunk_id in self._demod_map:
            try:
                self._demod_order.remove(chunk_id)
            except ValueError:
                pass

        # Store the demodulated signal block
        self._demod_map[chunk_id] = np.asarray(block, dtype=np.float32).reshape(-1)
        self._demod_order.append(chunk_id)
        self._evict_if_needed("demod")

        # Retrieve matching original block (if available)
        x = self._orig_map.get(chunk_id)
        y = self._demod_map.get(chunk_id)
        if x is None or y is None:
            # Cannot compute NCC until both are available
            return None

        # Compute NCC between the pair
        r = ncc_percent(x, y, eps=self.eps)
        if r is None:
            # If NCC failed (e.g., zero variance), remove the pair to avoid clutter
            self._orig_map.pop(chunk_id, None)
            self._demod_map.pop(chunk_id, None)
            try:
                self._orig_order.remove(chunk_id)
            except ValueError:
                pass
            try:
                self._demod_order.remove(chunk_id)
            except ValueError:
                pass
            return None

        # Successful result: store correlation info
        result = {
            "chunk_id": int(chunk_id),
            "ncc": float(r),                  # raw correlation in [-1,1]
            "percent": float(abs(r))  # absolute percentage correlation
        }

        # Clean up this pair from memory since it's already processed
        self._orig_map.pop(chunk_id, None)
        self._demod_map.pop(chunk_id, None)
        try:
            self._orig_order.remove(chunk_id)
        except ValueError:
            pass
        try:
            self._demod_order.remove(chunk_id)
        except ValueError:
            pass

        return result
    
    


def digital_accuracy(bits_ref: np.ndarray, bits_hat: np.ndarray) -> float:
    """
    Compute bit-level accuracy (%) between two binary sequences.

    Both arrays can represent bits per-sample or per-symbol.
    The function compares element-wise and returns the percentage
    of correctly matched bits.

    Parameters
    ----------
    bits_ref : np.ndarray
        Ground-truth or reference bit sequence (0s and 1s).
    bits_hat : np.ndarray
        Estimated or demodulated bit sequence.

    Returns
    -------
    accuracy : float
        Percentage of correctly matched bits (0–100).
    """
    # Ensure binary representation (threshold at 0.5)
    a = (bits_ref > 0.5).astype(np.uint8)
    b = (bits_hat > 0.5).astype(np.uint8)

    # Only compare up to the minimum common length
    n = min(len(a), len(b))
    if n == 0:
        return 0.0

    # Compute element-wise equality and convert to percentage
    return 100.0 * float(np.sum(a[:n] == b[:n])) / float(n)
