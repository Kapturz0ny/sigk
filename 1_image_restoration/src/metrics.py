import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class MetricsEvaluator:
    """Evaluates PSNR, SSIM, and LPIPS metrics for image pairs using TorchMetrics."""

    def __init__(self, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(self.device)

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Converts an HWC uint8 numpy array to an NCHW float tensor in [0, 1]."""
        tensor = torch.from_numpy(img).float().div(255.0)
        return tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def evaluate(self, img_true: np.ndarray, img_test: np.ndarray) -> dict[str, float]:
        """Calculates PSNR, SSIM, and LPIPS given two RGB images in uint8 format."""
        target = self._to_tensor(img_true)
        preds = self._to_tensor(img_test)

        return {
            "PSNR": self.psnr_metric(preds, target).item(),
            "SSIM": self.ssim_metric(preds, target).item(),
            "LPIPS": self.lpips_metric(preds, target).item(),
        }
