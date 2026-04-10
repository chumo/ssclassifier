import numpy as np

def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize 1D array of features to [0, 1] using min-max scaling per digit.
    This creates robustness to constant illumination differences.
    """
    f_min = features.min()
    f_max = features.max()
    if f_max > f_min:
        return (features - f_min) / (f_max - f_min)
    return np.zeros_like(features)
