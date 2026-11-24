import torch
import torch.nn.functional as F
from torch import nn

SCREEN_X_IDX = 10
SCREEN_Y_IDX = 11
APPARENT_RADIUS_IDX = 12


def weights_init(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ParamEmbedding(nn.Module):
    def __init__(self, param_dim: int, hidden: int, out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(param_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, params):
        return self.net(params)


def scene_geometry(params: torch.Tensor, size: int):
    """Mapa geometrii kuli z cech screen_x, screen_y, apparent_radius.

    Zwraca (grid_x, grid_y, maska) o kształcie [B, size, size]:
      - grid_x/grid_y w [-1, 1],
      - maska ~1 wewnątrz rzutu kuli, ~0 poza nią (miękka krawędź 1.2 px).
    """
    b = params.shape[0]
    device, dtype = params.device, params.dtype

    sx = params[:, SCREEN_X_IDX]
    sy = params[:, SCREEN_Y_IDX]
    ar = params[:, APPARENT_RADIUS_IDX]

    # środki pikseli: -1 + (i + 0.5) * 2 / size
    coords = (torch.arange(size, device=device, dtype=dtype) + 0.5) / size * 2 - 1
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")

    # NDC -> piksele (środek obiektu); promień w pikselach z NDC
    cx = (sx + 1.0) * 0.5 * size
    cy = (sy + 1.0) * 0.5 * size
    r = (ar + 1.0) * 0.25 * size

    px = (grid_x + 1.0) * 0.5 * size
    py = (grid_y + 1.0) * 0.5 * size
    dist = torch.sqrt(
        (px[None] - cx[:, None, None]) ** 2 + (py[None] - cy[:, None, None]) ** 2
    )

    mask = torch.sigmoid((r[:, None, None] - dist) / 1.2)
    # ar == -1 oznacza kulę za kamerą (sentryjnie) - maska musi być pusta
    visible = (ar > -0.999)[:, None, None].to(dtype)
    mask = mask * visible

    gx = grid_x[None].expand(b, -1, -1)
    gy = grid_y[None].expand(b, -1, -1)
    return gx, gy, mask


class DoubleConv(nn.Module):
    """2x (Conv 3x3 + GroupNorm + SiLU) - blok U-Neta."""

    def __init__(self, ch_in: int, ch_out: int, groups: int = 8):
        super().__init__()
        gn = min(groups, ch_out)
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False),
            nn.GroupNorm(gn, ch_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False),
            nn.GroupNorm(gn, ch_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetGenerator(nn.Module):
    def __init__(
        self,
        param_dim: int,
        z_dim: int,
        image_size: int = 128,
        base: int = 40,
        groups: int = 8,
    ):
        super().__init__()
        assert image_size % 16 == 0, "image_size musi być podzielne przez 16"
        self.param_dim = param_dim
        self.z_dim = z_dim
        self.image_size = image_size

        in_ch = param_dim + z_dim + 2 + 1  # parametry, z, x/y, maska
        self.inc = DoubleConv(in_ch, base, groups)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base * 2, groups))
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base * 2, base * 4, groups)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base * 4, base * 8, groups)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base * 8, base * 12, groups)
        )

        self.up4 = nn.Conv2d(base * 12, base * 8, 3, padding=1)
        self.conv4 = DoubleConv(base * 16, base * 8, groups)
        self.up3 = nn.Conv2d(base * 8, base * 4, 3, padding=1)
        self.conv3 = DoubleConv(base * 8, base * 4, groups)
        self.up2 = nn.Conv2d(base * 4, base * 2, 3, padding=1)
        self.conv2 = DoubleConv(base * 4, base * 2, groups)
        self.up1 = nn.Conv2d(base * 2, base, 3, padding=1)
        self.conv1 = DoubleConv(base * 2, base, groups)
        self.out = nn.Conv2d(base, 3, 1)

    def forward(self, z, params):
        b, size = params.shape[0], self.image_size
        gx, gy, mask = scene_geometry(params, size)
        pmap = params[:, :, None, None].expand(b, -1, size, size)
        if self.z_dim > 0:
            zmap = z[:, :, None, None].expand(b, -1, size, size)
            x = torch.cat([pmap, zmap, gx[:, None], gy[:, None], mask[:, None]], dim=1)
        else:
            x = torch.cat([pmap, gx[:, None], gy[:, None], mask[:, None]], dim=1)

        h1 = self.inc(x)
        h2 = self.down1(h1)
        h3 = self.down2(h2)
        h4 = self.down3(h3)
        h5 = self.down4(h4)

        u4 = F.interpolate(h5, scale_factor=2, mode="bilinear", align_corners=False)
        u4 = self.conv4(torch.cat([self.up4(u4), h4], dim=1))
        u3 = F.interpolate(u4, scale_factor=2, mode="bilinear", align_corners=False)
        u3 = self.conv3(torch.cat([self.up3(u3), h3], dim=1))
        u2 = F.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = self.conv2(torch.cat([self.up2(u2), h2], dim=1))
        u1 = F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = self.conv1(torch.cat([self.up1(u1), h1], dim=1))
        return torch.tanh(self.out(u1))


class Discriminator(nn.Module):
    """Krytyk WGAN: (obraz, parametry) -> skalar. Bez BatchNorm (WGAN-GP)."""

    def __init__(
        self,
        param_dim: int,
        image_size: int,
        channels: list,
        fc_dim: int,
        embed_hidden: int,
        embed_out: int,
    ):
        super().__init__()
        layers = [nn.Conv2d(3, channels[0], 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        for ch, ch2 in zip(channels, channels[1:]):
            layers += [
                nn.Conv2d(ch, ch2, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.features = nn.Sequential(*layers)
        feat_res = image_size // 2 ** len(channels)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1] * feat_res**2, fc_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.embed = ParamEmbedding(param_dim, embed_hidden, embed_out)
        assert fc_dim == embed_out, "fc_dim musi być równy embed_out (fuzja przez sumę)"
        self.out = nn.Linear(fc_dim, 1)

    def forward(self, img, params):
        h = self.classifier(self.features(img).flatten(1)) + self.embed(params)
        return self.out(torch.relu(h))
