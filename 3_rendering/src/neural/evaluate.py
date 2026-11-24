import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from flip_evaluator import evaluate

from .dataset import PhongSceneDataset
from .module import PhongGANModule


def compute_ssim(img0, img1):
    """SSIM kanałowo, obrazki [0,1] HWC float. Zwraca B."""
    from skimage.metrics import structural_similarity

    out = []
    for a, b in zip(img0, img1):
        vals = [
            structural_similarity(a[..., c], b[..., c], data_range=1.0)
            for c in range(a.shape[2])
        ]
        out.append(float(np.mean(vals)))
    return np.asarray(out)


def hausdorff_canny(a, b, low=50, high=150):
    """Odwrotna odległość Hausdorffa na krawędziach Canny. a,b: HWC [0,1] float."""
    import cv2

    ga = (a.mean(axis=2) * 255).astype(np.uint8)
    gb = (b.mean(axis=2) * 255).astype(np.uint8)
    ea = cv2.Canny(ga, low, high) > 0
    eb = cv2.Canny(gb, low, high) > 0
    if not ea.any() and not eb.any():
        return 0.0
    if not ea.any() or not eb.any():
        return float(np.hypot(*ea.shape))
    da = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 3)
    return float(max(da[eb].max(), db[ea].max()))


def compute_hausdorff(img0, img1):
    return np.asarray([hausdorff_canny(a, b) for a, b in zip(img0, img1)])


class Flipscore:
    """FLIP na obrazach [0,1] BCHW."""

    def __init__(self, device):
        self.device = device

    def __call__(self, t0, t1):
        """t*: B x 3 x H x W w [0,1]. Zwraca B (średnia FLIP po pikselach).""".
        out = []
        for ref, dist in zip(
            t0.detach().cpu().permute(0, 2, 3, 1),
            t1.detach().cpu().permute(0, 2, 3, 1),
        ):
            _, mean_error, _ = evaluate(
                np.ascontiguousarray(ref.numpy(), dtype=np.float32),
                np.ascontiguousarray(dist.numpy(), dtype=np.float32),
                "LDR",
                inputsRGB=True,
                applyMagma=False,
            )
            out.append(mean_error)
        return np.asarray(out)


class LPIPSScore:
    def __init__(self, device):
        import lpips

        self.model = lpips.LPIPS(net="alex").to(device).eval()
        self.device = device

    def __call__(self, t0, t1):
        with torch.no_grad():
            val = self.model(t0.to(self.device) * 2 - 1, t1.to(self.device) * 2 - 1)
        return val.flatten().cpu().numpy()


@torch.no_grad()
def generate_images(
    generator, dataset, indices, device, batch_size=64, z_mode="random"
):
    generator.eval()
    loader = DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    out = []
    for real, params, idx in loader:
        params = params.to(device)
        if generator.z_dim == 0 or z_mode == "zero":
            z = torch.zeros(params.size(0), generator.z_dim, device=device)
        else:
            z = torch.randn(params.size(0), generator.z_dim, device=device)
        fake = generator(z, params)
        out.append(((fake + 1) / 2).clamp(0, 1).cpu())
    return torch.cat(out)


def summarize(name, values):
    values = np.asarray(values, dtype=float)
    return {"metoda": name, "śr": float(values.mean()), "std": float(values.std())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="../dataset")
    parser.add_argument(
        "--checkpoint", default="../models/phong_gan/checkpoints/last.ckpt"
    )
    parser.add_argument(
        "--config",
        default="../configs/phong_gan.yaml",
        help="fallback, gdy checkpoint nie zawiera hyperparametrów Lightning",
    )
    parser.add_argument("--output_dir", default="../models/phong_gan/eval")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--skip", default="", help='metryki do pominięcia, np. "flip,lpips"'
    )
    args = parser.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    out_dir = Path(args.output_dir)
    (out_dir / "generated").mkdir(parents=True, exist_ok=True)
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    dataset = PhongSceneDataset(args.data_dir, split="test")
    print(f"Zbiór testowy: {len(dataset)} obrazów")

    try:
        module = PhongGANModule.load_from_checkpoint(
            args.checkpoint, map_location=device
        )
    except Exception:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        module = PhongGANModule.load_from_checkpoint(
            args.checkpoint, cfg=cfg, map_location=device
        )
    generator = module.generator.to(device).eval()
    print(f"Wczytano generator z {args.checkpoint} (epoka {module.current_epoch})")

    fake_t = generate_images(
        generator,
        dataset,
        range(len(dataset)),
        device,
        batch_size=args.batch_size,
        z_mode=getattr(module, "z_mode", "random"),
    )  # [0,1] BCHW
    real_t = []
    for i in range(len(dataset)):
        arr = (
            np.asarray(
                Image.open(
                    Path(args.data_dir)
                    / "images"
                    / f"image_{dataset.indices[i]:05}.png"
                ),
                dtype=np.float32,
            )
            / 255
        )
        real_t.append(torch.from_numpy(arr).permute(2, 0, 1))
    real_t = torch.stack(real_t)

    for i in range(len(dataset)):
        img = (fake_t[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(out_dir / "generated" / f"image_{i:05}.png")

    n = 10
    grid = make_grid(torch.cat([real_t[:n], fake_t[:n]]), nrow=n, padding=2)
    save_image(grid, out_dir / "comparison_grid.png")

    fake_np = fake_t.permute(0, 2, 3, 1).numpy()
    real_np = real_t.permute(0, 2, 3, 1).numpy()

    rows, per_image = [], {}
    metric_fns = {}
    if "ssim" not in skip:
        metric_fns["SSIM"] = compute_ssim(real_np, fake_np)
    if "hausdorff" not in skip:
        metric_fns["Hausdorff"] = compute_hausdorff(real_np, fake_np)

    black = np.zeros_like(fake_np)
    if "ssim" not in skip:
        metric_fns["SSIM_baseline"] = compute_ssim(real_np, black)
    if "hausdorff" not in skip:
        metric_fns["Hausdorff_baseline"] = compute_hausdorff(real_np, black)

    if "flip" not in skip:
        try:
            flip = Flipscore(device)
            metric_fns["FLIP"] = flip(real_t, fake_t)
            metric_fns["FLIP_baseline"] = flip(real_t, torch.zeros_like(fake_t))
        except Exception as exc:  # noqa: BLE001
            print(f"POMINIĘTO FLIP: {exc}")
    if "lpips" not in skip:
        try:
            lp = LPIPSScore(device)
            metric_fns["LPIPS"] = lp(real_t, fake_t)
            metric_fns["LPIPS_baseline"] = lp(real_t, torch.zeros_like(fake_t))
        except Exception as exc:  # noqa: BLE001
            print(f"POMINIĘTO LPIPS: {exc}")

    for name, values in metric_fns.items():
        per_image[name] = np.asarray(values).tolist()
        base = "_baseline" in name
        rows.append(summarize(name, values) | {"bazowa": base})

    with open(out_dir / "per_image_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_image, f)

    areas = (real_np.max(axis=3) > 0.01).reshape(len(real_np), -1).sum(axis=1)
    groups = [
        ("wszystkie", np.ones(len(areas), dtype=bool)),
        ("z obiektem", areas > 0),
        ("obiekt >= 100 px", areas >= 100),
        ("obiekt >= 300 px", areas >= 300),
        ("czarne (brak obiektu)", areas == 0),
    ]
    metric_names = ["FLIP", "LPIPS", "SSIM", "Hausdorff"]
    area_metrics = {}
    lines = [
        "| Grupa | n | FLIP | LPIPS | SSIM | Hausdorff |",
        "|---|---|---|---|---|---|",
    ]
    for label, mask in groups:
        row = {"n": int(mask.sum())}
        for name in metric_names:
            if name in per_image and mask.any():
                row[name] = float(np.asarray(per_image[name])[mask].mean())
        area_metrics[label] = row
        cells = []
        for name in metric_names:
            cells.append(f"{row[name]:.4f}" if name in row else "-")
        lines.append(f"| {label} | {row['n']} | " + " | ".join(cells) + " |")
    area_table = "\n".join(lines) + "\n"
    (out_dir / "results_by_area.md").write_text(area_table, encoding="utf-8")
    with open(out_dir / "per_area_metrics.json", "w", encoding="utf-8") as f:
        json.dump(area_metrics, f, indent=2)
    print("\nMetryki wg rozmiaru obiektu:\n" + area_table)

    def get(name):
        r = [r for r in rows if r["metoda"] == name]
        return f"{r[0]['śr']:.4f} ± {r[0]['std']:.4f}" if r else "-"

    table = (
        "| Metoda | FLIP | LPIPS | SSIM | Hausdorff |\n"
        "|---|---|---|---|---|\n"
        f"| neuronowy renderer | {get('FLIP')} | {get('LPIPS')} | {get('SSIM')} | {get('Hausdorff')} |\n"
        f"| bazowa (czarny obraz) | {get('FLIP_baseline')} | {get('LPIPS_baseline')} "
        f"| {get('SSIM_baseline')} | {get('Hausdorff_baseline')} |\n"
    )
    (out_dir / "results_table.md").write_text(table, encoding="utf-8")
    with open(out_dir / "results_summary.csv", "w", encoding="utf-8", newline="") as f:
        f.write("metoda,srednia,odchylenie_std\n")
        f.writelines(f"{r['metoda']},{r['śr']},{r['std']}\n" for r in rows)

    print("\n" + table)
    print(f"Wyniki zapisane w: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
