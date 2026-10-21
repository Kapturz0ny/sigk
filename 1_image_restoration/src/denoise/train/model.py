import torch
import torch.nn as nn


class SimpleGate(nn.Module):
    """Splits tensor along channels and multiplies the two halves element-wise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies nonlinear gating without traditional activation functions."""
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Core building block of NAFNet using built-in GroupNorm and Simplified Channel Attention."""

    def __init__(self, c: int) -> None:
        super().__init__()
        dw_channel = c * 2

        self.spatial_branch = nn.Sequential(
            nn.GroupNorm(1, c),
            nn.Conv2d(c, dw_channel, kernel_size=1),
            nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel),
            SimpleGate(),
        )
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1),
        )
        self.spatial_proj = nn.Conv2d(c, c, kernel_size=1)

        self.channel_branch = nn.Sequential(
            nn.GroupNorm(1, c),
            nn.Conv2d(c, c * 2, kernel_size=1),
            SimpleGate(),
            nn.Conv2d(c, c, kernel_size=1),
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes spatial and channel mixing with learnable residual scaling."""
        h = self.spatial_branch(x)
        h = self.spatial_proj(h * self.sca(h))
        y = x + h * self.beta

        return y + self.channel_branch(y) * self.gamma


class NAFNet(nn.Module):
    """U-Net shaped Nonlinear Activation Free Network for 256x256 image restoration."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        enc_blk_nums = [2, 2, 4, 8]
        dec_blk_nums = [2, 2, 2, 2]
        middle_blk_num = 4

        self.intro = nn.Conv2d(3, width, kernel_size=3, padding=1)
        self.ending = nn.Conv2d(width, 3, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Performs forward restoration pass with global residual connection."""
        x = self.intro(inp)
        skips = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x) + skip
            x = decoder(x)

        return inp + self.ending(x)
