import numpy as np
'''
class SignalMetrics:
    @staticmethod
    def normalized_cross_correlation(original: np.ndarray, demodulated: np.ndarray) -> float:
        """
        Computes the Normalized Cross-Correlation (NCC) between the original and demodulated signals.
        Returns a value between -1 and 1.
        """
        if original.shape != demodulated.shape:
            raise ValueError("Signals must have the same shape.")
        original_mean = np.mean(original)
        demodulated_mean = np.mean(demodulated)
        numerator = np.sum((original - original_mean) * (demodulated - demodulated_mean))
        denominator = np.sqrt(np.sum((original - original_mean) ** 2) * np.sum((demodulated - demodulated_mean) ** 2))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def compare_signals(original: np.ndarray, demodulated: np.ndarray) -> dict:
        """
        Returns a dictionary with NCC and other comparison metrics.
        """
        ncc = SignalMetrics.normalized_cross_correlation(original, demodulated)
        return {'NCC': ncc}
        '''
