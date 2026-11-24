import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from .dataset import PhongSceneDataset
from .module import PhongGANModule

BACKGROUND = (18, 18, 22)
TEXT = (235, 235, 240)
SUBTLE = (155, 155, 165)
ACCENT = (255, 190, 80)


def load_font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def load_module(checkpoint: str, config: str, device: str) -> PhongGANModule:
    try:
        return PhongGANModule.load_from_checkpoint(checkpoint, map_location=device)
    except Exception:  # noqa: BLE001 - checkpoint bez hyperparametrów
        cfg = yaml.safe_load(open(config))
        return PhongGANModule.load_from_checkpoint(
            checkpoint, cfg=cfg, map_location=device
        )


def foreground_area(image: torch.Tensor) -> int:
    """Liczba pikseli obiektu; image: [3, H, W] w [-1, 1]."""
    return int((image.max(dim=0).values > -0.98).sum())


@torch.no_grad()
def generate(module, dataset, positions, device, batch_size=32):
    """Predykcje dla wybranych pozycji w splicie; zwraca [N,3,H,W] w [0,1]."""
    module.generator.eval()
    reals, fakes = [], []
    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        batch = [dataset[i] for i in chunk]
        real = torch.stack([b[0] for b in batch])
        params = torch.stack([b[1] for b in batch]).to(device)
        z = module.sample_z(params.size(0), device)
        fake = module.generator(z, params)
        reals.append(((real + 1) / 2).clamp(0, 1))
        fakes.append(((fake.cpu() + 1) / 2).clamp(0, 1))
    return torch.cat(reals), torch.cat(fakes)


def pick_positions(dataset, args, rng):
    if args.indices:
        positions = list(args.indices)
        for p in positions:
            if not 0 <= p < len(dataset):
                raise SystemExit(f"Indeks {p} poza zakresem 0..{len(dataset) - 1}")
        return positions

    areas = []
    for i in range(len(dataset)):
        img, _, _ = dataset[i]
        areas.append(foreground_area(img))
    valid = [i for i in range(len(dataset)) if areas[i] >= args.min_area]

    if args.sort == "random":
        size = min(args.num, len(valid))
        return rng.choice(valid, size=size, replace=False).tolist()

    order = sorted(valid, key=lambda i: areas[i], reverse=True)
    return order[: args.num]


def make_figure(real, fake, dataset_idx, args):
    n = real.size(0)
    scale = args.scale
    tile = 128 * scale
    gap = 10
    margin = 18
    header = 66
    footer = 34
    label_w = 170
    caption_h = 30
    rows = 3 if args.diff else 2
    width = margin + label_w + n * (tile + gap) - gap + margin
    height = header + rows * (tile + caption_h) + footer

    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font_title = load_font(24)
    font_label = load_font(19)
    font_small = load_font(14)

    draw.text(
        (margin, 14),
        "Rendering neuronowy U-Net — referencja Phong vs predykcja modelu",
        fill=TEXT,
        font=font_title,
    )
    if args.indices:
        selection = "wybrane sceny"
    elif args.sort == "area":
        selection = "największe obiekty"
    else:
        selection = "losowe sceny"
    draw.text(
        (margin, 44),
        f"zbiór: {args.split}  ·  {n} scen  ·  {selection}",
        fill=SUBTLE,
        font=font_small,
    )

    row_names = ["referencja\n(renderer Phong)", "predykcja\n(model U-Net)"]
    if args.diff:
        row_names.append("różnica\n|ref − pred|")

    for r, name in enumerate(row_names):
        y = header + r * (tile + caption_h)
        draw.multiline_text(
            (margin, y + tile // 2 - 20),
            name,
            fill=ACCENT if r == 1 else TEXT,
            font=font_label,
            spacing=4,
        )

    for col in range(n):
        x = margin + label_w + col * (tile + gap)
        y_real = header

        real_img = Image.fromarray(
            (real[col].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        ).resize((tile, tile), Image.Resampling.LANCZOS)
        fake_img = Image.fromarray(
            (fake[col].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        ).resize((tile, tile), Image.Resampling.LANCZOS)

        canvas.paste(real_img, (x, y_real))
        canvas.paste(fake_img, (x, y_real + tile + caption_h))

        if args.diff:
            diff = (real[col] - fake[col]).abs().mean(dim=0, keepdim=True)
            diff = (diff * 4.0).clamp(0, 1).repeat(3, 1, 1)
            diff_img = Image.fromarray(
                (diff.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            ).resize((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(diff_img, (x, y_real + 2 * (tile + caption_h)))

        mask = real[col].max(dim=0).values > 0.01
        diff_fg = (
            (real[col] - fake[col]).abs().mean(dim=0)[mask].mean().item()
            if mask.any()
            else 0.0
        )
        area = int(mask.sum())
        caption = f"#{dataset_idx[col]:03d} · {area} px · L1fg {diff_fg:.3f}"
        tw = draw.textbbox((0, 0), caption, font=font_small)[2]
        draw.text(
            (x + (tile - tw) / 2, y_real + tile + 4),
            caption,
            fill=SUBTLE,
            font=font_small,
        )

    note = "L1fg — średni błąd bezwzględny wnętrza obiektu (0 = idealnie)."
    if args.diff:
        note += " Różnica wzmocniona 4×."
    draw.text(
        (margin, height - footer + 6),
        note,
        fill=SUBTLE,
        font=font_small,
    )
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="../models/phong_gan/checkpoints/last.ckpt"
    )
    parser.add_argument(
        "--config",
        default="../configs/phong_gan.yaml",
        help="fallback, gdy checkpoint nie zawiera hyperparametrów Lightning",
    )
    parser.add_argument("--data_dir", default="../dataset")
    parser.add_argument("--output", default="../models/phong_gan/visualization.png")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num", type=int, default=12)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="indeksy scen w wybranym splicie (kolejność jak podana)",
    )
    parser.add_argument("--sort", default="area", choices=["area", "random"])
    parser.add_argument(
        "--min_area",
        type=int,
        default=1,
        help="minimalne pole obiektu w pikselach (0 dopuszcza czarne sceny)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument(
        "--diff", action="store_true", help="dodaj trzeci rząd z mapą różnic"
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    module = load_module(args.checkpoint, args.config, device)
    module.generator.to(device)
    dataset = PhongSceneDataset(args.data_dir, split=args.split)

    rng = np.random.default_rng(args.seed)
    positions = pick_positions(dataset, args, rng)
    if not positions:
        raise SystemExit("Brak scen do wizualizacji.")

    real, fake = generate(module, dataset, positions, device)
    dataset_idx = [int(dataset.indices[p]) for p in positions]
    figure = make_figure(real, fake, dataset_idx, args)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.save(out)
    print(f"Zapisano wizualizację ({len(positions)} scen): {out.resolve()}")
    for pos, idx, r, f in zip(positions, dataset_idx, real, fake):
        mask = r.max(dim=0).values > 0.01
        l1 = (r - f).abs().mean(dim=0)[mask].mean().item() if mask.any() else 0.0
        print(f"  split[{pos:03d}] dataset[{idx:05d}]  L1fg={l1:.4f}")


if __name__ == "__main__":
    main()
