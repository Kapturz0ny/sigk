import numpy as np
import torch
import torch.nn as nn
import lightning as L
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.denoise.train.model import NAFNet


class NAFNetLitModule(L.LightningModule):
    """LightningModule with NAFNet, built-in losses, and inference."""

    def __init__(self, width: int = 32, lr: float = 1e-3, max_epochs: int = 100) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = NAFNet(width=width)
        self.pixel_loss = nn.SmoothL1Loss(beta=0.03)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the forward pass and clamps output to valid image range [0, 1]."""
        return torch.clamp(self.model(x), 0.0, 1.0)

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes weighted sum of SmoothL1, SSIM, and LPIPS losses in float32."""
        pred_f, target_f = pred.float(), target.float()
        l1 = self.pixel_loss(pred_f, target_f)
        ssim_loss = 1.0 - structural_similarity_index_measure(pred_f, target_f, data_range=1.0)
        lpips_loss = self.lpips(pred_f, target_f)
        return l1 + 0.1 * ssim_loss + 0.05 * lpips_loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Executes a single training step and logs the total loss."""
        noisy, clean = batch
        pred = self(noisy)
        loss = self._compute_loss(pred, clean)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        """Resets LPIPS state before validation to clear accumulated training scores."""
        self.lpips.reset()

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Evaluates model on validation split in float32 and logs PSNR, SSIM, and LPIPS."""
        noisy, clean = batch
        pred_f = self(noisy).float()
        clean_f = clean.float()

        self.log("val_psnr", self.psnr(pred_f, clean_f), prog_bar=True)
        self.log("val_ssim", self.ssim(pred_f, clean_f), prog_bar=True)
        self.log("val_lpips", self.lpips(pred_f, clean_f), prog_bar=True)

    def configure_optimizers(self) -> dict[str, object]:
        """Configures AdamW optimizer (updating only NAFNet weights) and Cosine scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @torch.no_grad()
    def predict_image(self, image: np.ndarray) -> np.ndarray:
        """Runs inference on a single RGB uint8 image for the evaluation pipeline."""
        self.eval()
        tensor_in = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).div(255.0)
        tensor_out = self(tensor_in.to(self.device))
        img_out = tensor_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (img_out * 255.0).round().astype(np.uint8)
