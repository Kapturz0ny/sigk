import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.metrics import MetricsEvaluator
from src.super_resolution.baseline import opencv_bicubic_upscale_batch
from src.super_resolution.train.dataset import SRDataModule
from src.super_resolution.train.lit_module import NAFNetSRLitModule

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

METRIC_NAMES = ("PSNR", "SSIM", "SNR", "LPIPS")
BASELINE_METHOD = "opencv_bicubic"
MODEL_METHOD = "nafnet_sr"


def load_config(config_path: Path) -> dict[str, Any]:
    """Loads an SR training configuration from a YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_device(device: str | None) -> torch.device:
    """Returns the requested device or the best available one."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Converts a CHW float tensor in [0, 1] to an HWC uint8 numpy array."""
    return (
        tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def load_model(weights_path: Path, device: torch.device) -> NAFNetSRLitModule:
    """Loads a weights-only Lightning checkpoint and prepares it for inference."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    model = NAFNetSRLitModule.load_from_checkpoint(weights_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def save_comparison_plot(
    lr_image: np.ndarray,
    bicubic_image: np.ndarray,
    sr_image: np.ndarray,
    hr_image: np.ndarray,
    image_metrics: dict[str, dict[str, float]],
    scale: int,
    output_path: Path,
) -> None:
    """Saves a four-panel LR / bicubic / NAFNet / HR comparison figure."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    panels = [
        (lr_image, f"LR input (x{scale})\n{lr_image.shape[0]}x{lr_image.shape[1]}"),
        (
            bicubic_image,
            "OpenCV bicubic\n"
            f"PSNR {image_metrics[BASELINE_METHOD]['PSNR']:.2f} dB, "
            f"SSIM {image_metrics[BASELINE_METHOD]['SSIM']:.4f}",
        ),
        (
            sr_image,
            "NAFNet SR\n"
            f"PSNR {image_metrics[MODEL_METHOD]['PSNR']:.2f} dB, "
            f"SSIM {image_metrics[MODEL_METHOD]['SSIM']:.4f}",
        ),
        (hr_image, f"Ground truth HR\n{hr_image.shape[0]}x{hr_image.shape[1]}"),
    ]

    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate_model(
    model: NAFNetSRLitModule,
    datamodule: SRDataModule,
    scale: int,
    device: torch.device,
    output_dir: Path,
    lpips_net_type: str = "alex",
    max_images: int | None = None,
    num_visuals: int = 4,
    metadata_extra: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluates NAFNet against the OpenCV bicubic baseline and saves metrics."""
    datamodule.setup()
    dataloader = datamodule.test_dataloader()
    image_paths = datamodule.test_paths

    evaluator = MetricsEvaluator(device=str(device), lpips_net_type=lpips_net_type)
    records: list[dict[str, Any]] = []
    processed = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = output_dir / "visuals"

    for lr_batch, hr_batch in tqdm(dataloader, desc="Evaluating"):
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)

        sr_batch = torch.clamp(model(lr_batch), 0.0, 1.0)
        bicubic_batch = opencv_bicubic_upscale_batch(lr_batch, scale)

        for index in range(lr_batch.shape[0]):
            if max_images is not None and processed >= max_images:
                break

            image_name = image_paths[processed].name
            hr_image = tensor_to_uint8(hr_batch[index])
            lr_image = tensor_to_uint8(lr_batch[index])
            bicubic_image = tensor_to_uint8(bicubic_batch[index])
            sr_image = tensor_to_uint8(sr_batch[index])

            image_metrics: dict[str, dict[str, float]] = {}
            for method, prediction in (
                (BASELINE_METHOD, bicubic_image),
                (MODEL_METHOD, sr_image),
            ):
                metrics = evaluator.evaluate(hr_image, prediction)
                evaluator.reset()
                image_metrics[method] = metrics
                records.append(
                    {
                        "image_name": image_name,
                        "scale": scale,
                        "method": method,
                        **metrics,
                    }
                )

            if processed < num_visuals:
                save_comparison_plot(
                    lr_image=lr_image,
                    bicubic_image=bicubic_image,
                    sr_image=sr_image,
                    hr_image=hr_image,
                    image_metrics=image_metrics,
                    scale=scale,
                    output_path=visuals_dir / f"{Path(image_name).stem}.png",
                )

            processed += 1

        if max_images is not None and processed >= max_images:
            break

    if not records:
        raise RuntimeError("No images were evaluated. Check the data directories.")

    per_image = pd.DataFrame(records)
    per_image.to_csv(output_dir / "per_image.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for method in (BASELINE_METHOD, MODEL_METHOD):
        method_df = per_image[per_image["method"] == method]
        row: dict[str, Any] = {
            "scale": scale,
            "method": method,
            "n_images": len(method_df),
        }
        for metric in METRIC_NAMES:
            row[f"{metric}_mean"] = method_df[metric].mean()
            row[f"{metric}_std"] = method_df[metric].std(ddof=0)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)

    comparison_rows: list[dict[str, Any]] = []
    for metric in METRIC_NAMES:
        baseline_value = summary.loc[
            summary["method"] == BASELINE_METHOD, f"{metric}_mean"
        ].iloc[0]
        model_value = summary.loc[
            summary["method"] == MODEL_METHOD, f"{metric}_mean"
        ].iloc[0]
        delta = float(model_value - baseline_value)
        improvement = -delta if metric == "LPIPS" else delta

        if abs(improvement) < 1e-12:
            better_method = "tie"
        elif improvement > 0:
            better_method = MODEL_METHOD
        else:
            better_method = BASELINE_METHOD

        comparison_rows.append(
            {
                "scale": scale,
                "metric": metric,
                "opencv_bicubic": float(baseline_value),
                "nafnet_sr": float(model_value),
                "delta": delta,
                "improvement": improvement,
                "better_method": better_method,
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "comparison.csv", index=False)

    metadata = {
        "scale": scale,
        "num_images": processed,
        "lpips_net_type": lpips_net_type,
        "summary": summary.to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
    }
    if metadata_extra is not None:
        metadata.update(metadata_extra)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=float)

    logger.info(f"Saved per-image metrics to {output_dir / 'per_image.csv'}")
    logger.info(f"Saved summary to {output_dir / 'summary.csv'}")
    logger.info(f"Saved comparison to {output_dir / 'comparison.csv'}")

    return per_image, summary, comparison


def main() -> None:
    """Parses arguments and compares a trained NAFNet SR model with OpenCV bicubic."""
    parser = argparse.ArgumentParser(
        description="Evaluate NAFNet super-resolution against OpenCV bicubic."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/super_resolution/train/config_scale4.yaml",
        help="Path to the YAML configuration used for training.",
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for metrics and visualizations (default: results/upscale/x<scale>).",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional limit on the number of evaluated images.",
    )
    parser.add_argument(
        "--num_visuals",
        type=int,
        default=4,
        help="Number of four-panel comparison images to save.",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device, e.g. cuda or cpu."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional evaluation batch-size override.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Optional evaluation worker-count override.",
    )
    parser.add_argument(
        "--lpips_net_type",
        type=str,
        default=None,
        choices=("alex", "vgg", "squeeze"),
        help="Optional LPIPS backbone override.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    scale = int(model_cfg["scale"])
    lpips_net_type = args.lpips_net_type or model_cfg.get("lpips_net_type", "alex")
    device = get_device(args.device)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("results/upscale") / f"x{scale}"
    )

    torch.set_float32_matmul_precision("high")
    model = load_model(Path(args.weights), device)
    if model.model.scale != scale:
        raise ValueError(
            f"Checkpoint scale ({model.model.scale}) does not match config scale ({scale})."
        )

    datamodule = SRDataModule(
        train_dir=Path(data_cfg["train_dir"]),
        test_dir=Path(data_cfg["test_dir"]),
        batch_size=args.batch_size or data_cfg["batch_size"],
        num_workers=args.num_workers
        if args.num_workers is not None
        else data_cfg["num_workers"],
        hr_size=data_cfg["hr_size"],
        scale=scale,
        val_split=data_cfg["val_split"],
        seed=cfg["seed"],
    )

    logger.info(f"Evaluating {args.weights} (scale=x{scale})...")
    _, summary, comparison = evaluate_model(
        model=model,
        datamodule=datamodule,
        scale=scale,
        device=device,
        output_dir=output_dir,
        lpips_net_type=lpips_net_type,
        max_images=args.max_images,
        num_visuals=args.num_visuals,
        metadata_extra={
            "checkpoint": str(Path(args.weights)),
            "config": str(config_path),
            "device": str(device),
        },
    )

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nComparison with OpenCV bicubic:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
