import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.baseline import apply_histogram_equalization
from src.metrics import MetricsEvaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

GT_DIR = Path("data/processed/test/gt")
INPUT_DIR = Path("data/processed/test/input")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models/hvi_cidnet")


def get_inference_fn(method_name: str, weights_path: str | None = None) -> Callable[[np.ndarray], np.ndarray] | None:
    """Returns the inference callable for the selected method ('input', 'baseline', 'hvi_cidnet')."""
    if method_name == "input":
        return None

    if method_name == "baseline":
        return apply_histogram_equalization

    if method_name == "hvi_cidnet":
        if not weights_path:
            raise ValueError("Weights path (--weights) is required for 'hvi_cidnet' evaluation.")
        import torch

        from src.train.lit_module import HVICIDNetLitModule

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = HVICIDNetLitModule.load_from_checkpoint(weights_path, map_location=device)
        model.to(device)
        model.eval()
        return model.predict_image

    raise ValueError(f"Unknown method: {method_name}. Choose from: input, baseline, hvi_cidnet.")


def evaluate_method(method_name: str, weights_path: str | None = None) -> None:
    """Evaluates the chosen method on the test set and saves outputs named after the method."""
    inference_fn = get_inference_fn(method_name, weights_path)

    results_csv = RESULTS_DIR / f"{method_name}.csv"
    output_dir = None if method_name == "input" else Path(f"data/processed/test/{method_name}")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = MetricsEvaluator()
    image_paths = sorted(INPUT_DIR.glob("*_input.png"))
    metrics_records = []

    for img_path in tqdm(image_paths, desc=f"Evaluating [{method_name}]"):
        base_name = img_path.name.replace("_input.png", "")
        gt_path = GT_DIR / f"{base_name}_gt.png"

        gt_img = imageio.imread(gt_path)
        input_img = imageio.imread(img_path)

        processed_img = inference_fn(input_img) if inference_fn is not None else input_img

        if output_dir is not None:
            imageio.imsave(output_dir / f"{base_name}.png", processed_img)

        metrics = evaluator.evaluate(gt_img, processed_img)
        metrics["image_name"] = base_name
        metrics_records.append(metrics)

    df_metrics = pd.DataFrame(metrics_records)[["image_name", "PSNR", "SSIM", "LPIPS"]]
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(results_csv, index=False)

    logging.info(
        f"[{method_name}] Saved to {results_csv} | "
        f"PSNR: {df_metrics['PSNR'].mean():.2f} | "
        f"SSIM: {df_metrics['SSIM'].mean():.4f} | "
        f"LPIPS: {df_metrics['LPIPS'].mean():.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Low-Light Enhancement on test set.")
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default="baseline",
        choices=["input", "baseline", "hvi_cidnet"],
        help="Evaluation method (automatically names output folder and CSV).",
    )
    parser.add_argument("--weights", "-w", type=str, default=None, help="Optional checkpoint path.")

    args = parser.parse_args()
    evaluate_method(args.method, args.weights)


if __name__ == "__main__":
    main()
