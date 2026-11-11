from math import pi

import torch
import torch.nn as nn


class RGB_HVI(nn.Module):
    """RGB <-> HVI color space transform."""

    def __init__(self) -> None:
        super().__init__()
        self.density_k = torch.nn.Parameter(torch.full([1], 0.2))
        self.gated = False
        self.gated2 = False
        self.alpha = 1.0
        self.alpha_s = 1.3

    def HVIT(self, img: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        img = img.clamp(0.0, 1.0)
        device, dtypes = img.device, img.dtype

        hue = torch.zeros(img.shape[0], img.shape[2], img.shape[3], device=device, dtype=dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)

        hue[img[:, 2] == value] = 4.0 + ((img[:, 0] - img[:, 1]) / (value - img_min + eps))[img[:, 2] == value]
        hue[img[:, 1] == value] = 2.0 + ((img[:, 2] - img[:, 0]) / (value - img_min + eps))[img[:, 1] == value]
        hue[img[:, 0] == value] = (0.0 + ((img[:, 1] - img[:, 2]) / (value - img_min + eps))[img[:, 0] == value]) % 6

        hue[img_min == value] = 0.0
        hue = hue / 6.0

        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        k = self.density_k.clamp(min=0.05, max=2.0)
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value
        return torch.cat([H, V, I], dim=1)

    def PHVIT(self, img: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        H, V, I = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

        H = torch.clamp(H, -1.0, 1.0)
        V = torch.clamp(V, -1.0, 1.0)
        I = torch.clamp(I, 0.0, 1.0)

        v = I
        k = self.density_k.clamp(min=0.05, max=2.0)
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = (H / (color_sensitive + eps)).clamp(-1.0, 1.0)
        V = (V / (color_sensitive + eps)).clamp(-1.0, 1.0)

        h = (torch.atan2(V + eps, H + eps) / (2 * pi)) % 1.0
        s = torch.sqrt(H**2 + V**2 + eps)

        if self.gated:
            s = s * self.alpha_s

        s = torch.clamp(s, 0.0, 1.0)
        v = torch.clamp(v, 0.0, 1.0)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1.0 - s)
        q = v * (1.0 - (f * s))
        t = v * (1.0 - ((1.0 - f) * s))

        for idx, (rv, gv, bv) in enumerate([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]):
            mask = hi == idx
            r[mask], g[mask], b[mask] = rv[mask], gv[mask], bv[mask]

        rgb = torch.stack([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return torch.clamp(rgb, 0.0, 1.0)

    hvit = HVIT
    phvit = PHVIT
