import torch
import torch.nn.functional as F
from torch import nn

from src.denoise.train.model import SimpleGate


class LayerNorm2d(nn.Module):
    """Layer normalization over channel, height, and width with per-channel affine weights."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes each sample over all non-batch dimensions."""
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class NAFBlock(nn.Module):
    """Nonlinear-activation-free block used by the super-resolution NAFNet body."""

    def __init__(self, channels: int, expand_ratio: int = 2) -> None:
        super().__init__()
        mid_channels = channels * expand_ratio

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1)

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, mid_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=1)

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the spatial branch with channel attention followed by the FFN branch."""
        out = self.norm1(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.sg(out)
        out = out * self.sca(out)
        out = self.conv3(out)
        x = x + out * self.beta

        out = self.norm2(x)
        out = self.conv4(out)
        out = self.sg(out)
        out = self.conv5(out)
        return x + out * self.gamma


class NAFNetSR(nn.Module):
    """Single-scale NAFNet with a PixelShuffle upsampling head and bicubic residual base."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 64,
        scale: int = 4,
        num_blocks: int = 32,
    ) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError(f"scale must be at least 1, got {scale}.")

        self.scale = scale
        self.intro = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[NAFBlock(embed_dim) for _ in range(num_blocks)])
        self.body_tail = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        up_layers: list[nn.Module] = []
        current_scale = scale
        while current_scale > 1:
            step = 2 if current_scale % 2 == 0 else current_scale
            up_layers.append(
                nn.Conv2d(embed_dim, embed_dim * (step**2), kernel_size=3, padding=1)
            )
            up_layers.append(nn.PixelShuffle(step))
            current_scale //= step

            if current_scale > 1:
                up_layers.append(nn.LeakyReLU(0.2, inplace=True))

        up_layers.append(nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1))
        self.up = nn.Sequential(*up_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restores and upsamples the input by the configured scale factor."""
        features = self.intro(x)
        residual = self.body(features)
        features = features + self.body_tail(residual)

        base = F.interpolate(
            x, scale_factor=self.scale, mode="bilinear", align_corners=False
        )
        return self.up(features) + base
