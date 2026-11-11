import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image import image_gradients, structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.train.model import HVICIDNet


class HVICIDNetLitModule(L.LightningModule):
    """LightningModule wrapping HVI-CIDNet with Dual-Space Loss (sRGB + HVI) and inference."""

    def __init__(
        self,
        channels: list[int] | None = None,
        heads: list[int] | None = None,
        lr: float = 4e-4,
        max_epochs: int = 100,
        hvi_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = HVICIDNet(channels=channels, heads=heads)
        self.pixel_loss = nn.SmoothL1Loss(beta=0.01)

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.model(x), 0.0, 1.0)

    @staticmethod
    def _edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dy_pred, dx_pred = image_gradients(pred)
        dy_target, dx_target = image_gradients(target)
        return F.l1_loss(dy_pred, dy_target) + F.l1_loss(dx_pred, dx_target)

    def _compute_loss(
        self,
        rgb_pred: torch.Tensor,
        hvi_pred: torch.Tensor,
        rgb_target: torch.Tensor,
    ) -> torch.Tensor:
        rgb_pred_f, hvi_pred_f, rgb_target_f = rgb_pred.float(), hvi_pred.float(), rgb_target.float()
        hvi_target_f = self.model.trans.HVIT(rgb_target_f)

        loss_rgb = (
            self.pixel_loss(rgb_pred_f, rgb_target_f)
            + 0.1 * self._edge_loss(rgb_pred_f, rgb_target_f)
            + 0.2 * (1.0 - structural_similarity_index_measure(rgb_pred_f, rgb_target_f, data_range=1.0))
            + 0.1 * self.lpips(rgb_pred_f, rgb_target_f)
        )
        loss_hvi = self.pixel_loss(hvi_pred_f, hvi_target_f) + 0.1 * self._edge_loss(hvi_pred_f, hvi_target_f)
        return loss_rgb + self.hparams.hvi_weight * loss_hvi

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        dark, clean = batch
        rgb_pred, hvi_pred = self.model.forward_features(dark)
        loss = self._compute_loss(rgb_pred, hvi_pred, clean)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.lpips.reset()

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        dark, clean = batch
        pred_f, clean_f = self(dark).float(), clean.float()

        self.log("val_psnr", self.psnr(pred_f, clean_f), prog_bar=True)
        self.log("val_ssim", self.ssim(pred_f, clean_f), prog_bar=True)
        self.log("val_lpips", self.lpips(pred_f, clean_f), prog_bar=True)

    def configure_optimizers(self) -> dict[str, object]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=1e-7)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @torch.no_grad()
    def predict_image(self, image: np.ndarray) -> np.ndarray:
        self.eval()
        tensor_in = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).div(255.0)
        tensor_out = self(tensor_in.to(self.device))
        img_out = tensor_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (img_out * 255.0).round().astype(np.uint8)
