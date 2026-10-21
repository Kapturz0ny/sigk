import numpy as np
from skimage.restoration import denoise_bilateral


def apply_bilateral_filter(
    image: np.ndarray,
    sigma_color: float = 0.025,
    sigma_spatial: float = 1.5,
) -> np.ndarray:
    """Applies the skimage bilateral denoise filter with configurable parameters."""
    image_normalized = image.astype(np.float32) / 255.0

    denoised = denoise_bilateral(
        image_normalized,
        channel_axis=-1,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
    )

    return (np.clip(denoised, 0.0, 1.0) * 255).astype(np.uint8)
