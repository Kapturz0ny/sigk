import lightning as L
import numpy as np
import torch
from torch import nn
from torchmetrics.audio import SignalNoiseRatio
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.super_resolution.train.model import NAFNetSR


class NAFNetSRLitModule(L.LightningModule):
    """LightningModule for NAFNet super-resolution with combined pixel, SSIM, and LPIPS loss."""

    def __init__(
        self,
        model_kwargs: dict[str, int] | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
        max_epochs: int = 100,
        lpips_net_type: str = "alex",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if model_kwargs is None:
            model_kwargs = {"embed_dim": 64, "scale": 4, "num_blocks": 32}

        self.model = NAFNetSR(**model_kwargs)
        self.pixel_loss = nn.SmoothL1Loss(beta=0.03)

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.snr = SignalNoiseRatio()
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=lpips_net_type, normalize=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the super-resolution forward pass."""
        return self.model(x)

    def _reset_metrics(self) -> None:
        """Clears metric states so each validation/test epoch starts fresh."""
        self.psnr.reset()
        self.ssim.reset()
        self.snr.reset()
        self.lpips.reset()

    def _lpips_value(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes a differentiable LPIPS value without keeping its graph in metric state."""
        value = self.lpips(pred, target)
        self.lpips.reset()
        return value

    def _compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lpips_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes SmoothL1 + weighted SSIM and LPIPS losses in float32."""
        pred_f = pred.float()
        target_f = target.float()
        pred_clamped = torch.clamp(pred_f, 0.0, 1.0)

        pixel_loss = self.pixel_loss(pred_f, target_f)
        ssim_loss = 1.0 - structural_similarity_index_measure(
            pred_clamped, target_f, data_range=1.0
        )
        if lpips_value is None:
            lpips_value = self._lpips_value(pred_clamped, target_f)

        return pixel_loss + 0.1 * ssim_loss + 0.05 * lpips_value

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Executes a single training step and logs the total loss."""
        lr_image, hr_image = batch
        sr_image = self(lr_image)
        loss = self._compute_loss(sr_image, hr_image)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        """Resets metric states before validation."""
        self._reset_metrics()

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Evaluates the model on the validation split and logs losses and metrics."""
        lr_image, hr_image = batch
        sr_image = self(lr_image)
        sr_image_clamped = torch.clamp(sr_image, 0.0, 1.0)

        lpips_value = self._lpips_value(sr_image_clamped, hr_image)
        loss = self._compute_loss(sr_image, hr_image, lpips_value=lpips_value)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log(
            "val_psnr",
            self.psnr(sr_image_clamped, hr_image),
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_ssim",
            self.ssim(sr_image_clamped, hr_image),
            on_epoch=True,
            prog_bar=True,
        )
        self.log("val_lpips", lpips_value.detach(), on_epoch=True)

    def on_test_epoch_start(self) -> None:
        """Resets metric states before testing."""
        self._reset_metrics()

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Evaluates the model on the test split and logs all metrics."""
        lr_image, hr_image = batch
        sr_image = self(lr_image)
        sr_image_clamped = torch.clamp(sr_image, 0.0, 1.0)

        lpips_value = self._lpips_value(sr_image_clamped, hr_image)
        loss = self._compute_loss(sr_image, hr_image, lpips_value=lpips_value)

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_psnr", self.psnr(sr_image_clamped, hr_image), on_epoch=True)
        self.log("test_ssim", self.ssim(sr_image_clamped, hr_image), on_epoch=True)
        self.log("test_snr", self.snr(sr_image_clamped, hr_image), on_epoch=True)
        self.log("test_lpips", lpips_value.detach(), on_epoch=True)

    def configure_optimizers(self) -> dict[str, object]:
        """Configures AdamW and a step-based cosine learning-rate schedule."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.9),
        )
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @torch.no_grad()
    def predict_image(self, image: np.ndarray) -> np.ndarray:
        """Upscales a single RGB uint8 image and returns an RGB uint8 image."""
        self.eval()
        tensor_in = (
            torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).div(255.0)
        )
        tensor_out = torch.clamp(self(tensor_in.to(self.device)), 0.0, 1.0)
        image_out = tensor_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (image_out * 255.0).round().astype(np.uint8)
