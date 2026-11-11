import numpy as np


def _equalize_channel_2d(channel: np.ndarray) -> np.ndarray:
    """Applies standard 256-bin histogram equalization to a single 2D uint8 channel."""
    hist, _ = np.histogram(channel.ravel(), bins=256, range=(0, 256))
    cdf = hist.cumsum()

    non_zero = cdf > 0
    if not np.any(non_zero):
        return channel.copy()

    cdf_min = cdf[non_zero][0]
    total_pixels = cdf[-1]

    if total_pixels == cdf_min:
        return channel.copy()

    lut = np.round((cdf - cdf_min) * 255.0 / (total_pixels - cdf_min))
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[channel]


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Applies histogram equalization independently to each R, G, B channel."""
    return np.stack([_equalize_channel_2d(image[..., c]) for c in range(3)], axis=-1)
