import torch
import torch.nn as nn
import torch.nn.functional as F

from src.train.hvi_transform import RGB_HVI


def dw_conv(in_ch: int, out_ch: int, kernel_size: int = 3, bias: bool = False) -> nn.Sequential:
    """Pointwise 1x1 conv followed by Depthwise conv."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias),
        nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2, groups=out_ch, bias=bias),
    )


class HV_FFN(nn.Module):
    """Intensity Enhance Layer (IEL) with dual depthwise Tanh residual branches."""

    def __init__(self, dim: int, expansion: float = 2.66, bias: bool = False) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = dw_conv(dim, hidden * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.project_in(x).chunk(2, dim=1)
        x1 = self.tanh(self.dwconv1(x1)) + x1
        x2 = self.tanh(self.dwconv2(x2)) + x2
        return self.project_out(x1 * x2)


class CAB(nn.Module):
    """Cross-Attention Block"""

    def __init__(self, dim: int, num_heads: int, bias: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = dw_conv(dim, dim, bias=bias)
        self.kv = dw_conv(dim, dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.q(x)
        k, v = self.kv(y).chunk(2, dim=1)

        q = F.normalize(q.unflatten(1, (self.num_heads, -1)).flatten(3), dim=-1)
        k = F.normalize(k.unflatten(1, (self.num_heads, -1)).flatten(3), dim=-1)
        v = v.unflatten(1, (self.num_heads, -1)).flatten(3)

        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        return self.project_out((attn @ v).reshape(b, c, h, w))


class LCA(nn.Module):
    """Lighten Cross-Attention block with full pre-norm residual connections."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm_x = nn.GroupNorm(1, dim)
        self.norm_y = nn.GroupNorm(1, dim)
        self.norm_ffn = nn.GroupNorm(1, dim)
        self.cab = CAB(dim, num_heads)
        self.ffn = HV_FFN(dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x + self.cab(self.norm_x(x), self.norm_y(y))
        x = x + self.ffn(self.norm_ffn(x))
        return x


class RRB(nn.Module):
    """Region Refinement Block"""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.mid_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.ctx_pred = nn.Sequential(nn.Conv2d(dim, 2, 3, padding=1), nn.LeakyReLU(0.1, inplace=True))
        self.mask_pred = nn.Sequential(nn.Conv2d(dim + 2, dim, kernel_size=1), nn.Sigmoid())

        self.ms_convs = nn.ModuleList(
            [nn.Conv2d(dim, dim, kernel_size=k, padding=k // 2, groups=dim) for k in (5, 7, 9)]
        )
        self.ms_fuse = nn.Conv2d(dim * 3, dim, kernel_size=1)
        self.dyn_weight = nn.Sequential(nn.GroupNorm(1, dim), nn.Conv2d(dim, dim, 3, padding=1, groups=dim))

        self.attn_branch = CAB(dim, num_heads)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_m = self.mid_conv(self.norm(x))
        c_s = self.ctx_pred(c_m)

        m_o = self.mask_pred(torch.cat([c_m, c_s], dim=1))
        y_high = m_o * c_m
        y_low = (1.0 - m_o) * c_m

        f_ms = self.ms_fuse(torch.cat([conv(y_low) for conv in self.ms_convs], dim=1))
        out_low = f_ms * self.dyn_weight(f_ms)
        out_high = self.attn_branch(y_high, y_high)

        return x + self.out_proj(out_low + out_high)


class HVICIDNet(nn.Module):
    """Dual-branch HVI-CIDNet with PixelShuffle upsampling and RRB refinement."""

    def __init__(
        self,
        channels: list[int] | None = None,
        heads: list[int] | None = None,
    ) -> None:
        super().__init__()
        channels = channels or [36, 36, 72, 144]
        heads = heads or [1, 2, 4, 8]

        self.trans = RGB_HVI()
        self.hv_in = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(3, channels[0], 3, bias=False))
        self.i_in = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(1, channels[0], 3, bias=False))

        self.hv_downs = nn.ModuleList()
        self.i_downs = nn.ModuleList()
        self.hv_encs = nn.ModuleList()
        self.i_encs = nn.ModuleList()

        for idx in range(len(channels) - 1):
            c_in, c_out = channels[idx], channels[idx + 1]
            h = heads[idx + 1]
            self.hv_downs.append(nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False))
            self.i_downs.append(nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False))
            self.hv_encs.append(LCA(c_out, h))
            self.i_encs.append(LCA(c_out, h))

        self.hv_ups = nn.ModuleList()
        self.i_ups = nn.ModuleList()
        self.i_rrbs = nn.ModuleList()
        self.hv_decs = nn.ModuleList()
        self.i_decs = nn.ModuleList()

        for idx in reversed(range(len(channels) - 1)):
            c_in, c_out = channels[idx + 1], channels[idx]
            h = heads[idx + 1]
            self.i_rrbs.append(RRB(c_in, h))
            self.hv_decs.append(LCA(c_in, h))
            self.i_decs.append(LCA(c_in, h))
            self.hv_ups.append(nn.Sequential(nn.Conv2d(c_in, c_out * 4, 1, bias=False), nn.PixelShuffle(2)))
            self.i_ups.append(nn.Sequential(nn.Conv2d(c_in, c_out * 4, 1, bias=False), nn.PixelShuffle(2)))

        # Full-resolution refinement after fusing the shallowest skip connection
        self.hv_refine = nn.Sequential(nn.GroupNorm(1, channels[0]), HV_FFN(channels[0]))
        self.i_refine = nn.Sequential(nn.GroupNorm(1, channels[0]), HV_FFN(channels[0]))

        self.hv_out = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(channels[0], 2, 3))
        self.i_out = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(channels[0], 1, 3))

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = x.shape
        pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        hvi = self.trans.HVIT(x)
        i_raw = hvi[:, 2:3]

        hv, i_feat = self.hv_in(hvi), self.i_in(i_raw)
        skips: list[tuple[torch.Tensor, torch.Tensor]] = []

        for hv_down, i_down, hv_enc, i_enc in zip(self.hv_downs, self.i_downs, self.hv_encs, self.i_encs):
            skips.append((hv, i_feat))
            hv, i_feat = hv_down(hv), i_down(i_feat)
            hv, i_feat = hv_enc(hv, i_feat), i_enc(i_feat, hv)

        for rrb, hv_dec, i_dec, hv_up, i_up, (skip_hv, skip_i) in zip(
            self.i_rrbs, self.hv_decs, self.i_decs, self.hv_ups, self.i_ups, reversed(skips)
        ):
            mid = rrb(i_feat)
            hv, i_feat = hv_dec(hv, i_feat), i_dec(mid, hv)
            hv = hv_up(hv) + skip_hv
            i_feat = i_up(i_feat) + skip_i

        hv = hv + self.hv_refine(hv)
        i_feat = i_feat + self.i_refine(i_feat)

        hv_pred = hvi[:, 0:2] + self.hv_out(hv)
        i_pred = (i_raw + self.i_out(i_feat)).clamp(0.0, 1.0)

        output_hvi = torch.cat([hv_pred, i_pred], dim=1)
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb[:, :, :h, :w], output_hvi[:, :, :h, :w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb_pred, _ = self.forward_features(x)
        return rgb_pred
