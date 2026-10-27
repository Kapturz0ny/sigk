import cv2
import numpy as np
import torch


def opencv_bicubic_upscale_batch(
    lr_batch: torch.Tensor, scale_factor: int
) -> torch.Tensor:
    """Upscales a [B, C, H, W] tensor in [0, 1] using OpenCV bicubic interpolation."""
    if scale_factor < 1:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}.")

    device = lr_batch.device
    lr_np = lr_batch.detach().cpu().permute(0, 2, 3, 1).numpy()

    hr_list = []
    for img in lr_np:
        height, width = img.shape[:2]
        img_up = cv2.resize(
            img,
            (width * scale_factor, height * scale_factor),
            interpolation=cv2.INTER_CUBIC,
        )
        hr_list.append(np.clip(img_up, 0.0, 1.0))

    hr_np = np.stack(hr_list, axis=0)
    return torch.from_numpy(hr_np).permute(0, 3, 1, 2).float().to(device)
