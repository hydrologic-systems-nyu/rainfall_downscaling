import numpy as np
from scipy.ndimage import uniform_filter

def compute_fss(pred, target, threshold=0.5, window_size=9):
    """
    Compute FSS for a single pair of 2D maps using NumPy.

    Args:
        pred, target: np.ndarray of shape (H, W)
        threshold: float, binarization threshold
        window_size: int, sliding window size

    Returns:
        fss: float
    """
    pred_bin = (pred > threshold).astype(np.float32)
    target_bin = (target > threshold).astype(np.float32)

    pred_frac = uniform_filter(pred_bin, size=window_size, mode='constant')
    target_frac = uniform_filter(target_bin, size=window_size, mode='constant')

    numerator = np.mean((pred_frac - target_frac) ** 2)
    denominator = np.mean(pred_frac ** 2 + target_frac ** 2)

    fss = 1 - numerator / (denominator + 1e-8)
    return fss


def fss_batch(pred, target, threshold=1.0, window_size=5, reduction='mean'):
    """
    Compute FSS for batched data using NumPy.

    Args:
        pred, target: np.ndarray of shape (B, C, T, H, W)
        threshold: float, binarization threshold
        window_size: int, sliding window size
        reduction: 'mean' or 'none'

    Returns:
        float (mean FSS) or list of FSS values
    """
    B, C, T, H, W = pred.shape
    fss_list = []

    for b in range(B):
        for c in range(C):
            for t in range(T):
                fss = compute_fss(
                    pred[b, c, t], target[b, c, t],
                    threshold=threshold,
                    window_size=window_size
                )
                fss_list.append(fss)

    if reduction == 'mean':
        return float(np.mean(fss_list))
    else:
        return fss_list