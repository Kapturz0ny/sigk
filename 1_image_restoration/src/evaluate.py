import argparse
import logging
from functools import partial
from pathlib import Path
from typing import Callable
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.metrics import MetricsEvaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_inference_fn(
    method_name: str | None, weights_path: str | None = None
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Returns the appropriate inference function, or None for raw input evaluation."""
    if method_name is None:
        return None

    if method_name == "bilateral_001":
        from src.denoise.baseline import apply_bilateral_filter
        return partial(apply_bilateral_filter, sigma_color=0.025, sigma_spatial=1.5)

    if method_name == "bilateral_003":
        from src.denoise.baseline import apply_bilateral_filter
        return partial(apply_bilateral_filter, sigma_color=0.075, sigma_spatial=2.0)

    if method_name == "nafnet":
        if not weights_path:
            raise ValueError("Weights path (--weights) is required for nafnet evaluation.")
        import torch
        from src.denoise.train.lit_module import NAFNetLitModule

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = NAFNetLitModule.load_from_checkpoint(weights_path, map_location=device)
        model.to(device)
        model.eval()
        return model.predict_image

    raise ValueError(f"Unknown evaluation method: {method_name}")


def evaluate_dataset(
    gt_dir: Path,
    input_dir: Path,
    results_csv: Path,
    output_dir: Path | None = None,
    inference_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    """Evaluates a dataset (with optional inference) and saves metrics to CSV."""
    if not input_dir.exists() or not gt_dir.exists():
        logging.error("Input or Ground Truth directories do not exist.")
        return

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = MetricsEvaluator()
    image_paths = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))
    metrics_records = []

    logging.info(f"Starting evaluation on {len(image_paths)} images...")

    for img_path in tqdm(image_paths, desc="Processing images"):
        gt_img = cv2.cvtColor(cv2.imread(str(gt_dir / img_path.name)), cv2.COLOR_BGR2RGB)
        input_img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        processed_img = inference_fn(input_img) if inference_fn is not None else input_img

        if output_dir is not None:
            processed_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / img_path.name), processed_bgr)

        metrics = evaluator.evaluate(gt_img, processed_img)
        metrics["image_name"] = img_path.name
        metrics_records.append(metrics)

    df_metrics = pd.DataFrame(metrics_records)
    df_metrics = df_metrics[["image_name", "PSNR", "SSIM", "LPIPS"]]

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(results_csv, index=False)
    logging.info(f"Metrics saved successfully to {results_csv}")


def main() -> None:
    """Parses arguments and runs the universal evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Universal ML Evaluation Pipeline.")
    parser.add_argument("--gt_dir", "-gt", type=str, required=True, help="Path to Ground Truth images.")
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Path to input (noisy/low-res) images.")
    parser.add_argument("--results_csv", "-r", type=str, required=True, help="Path to save metrics CSV.")
    parser.add_argument("--output_dir", "-o", type=str, default=None, help="Optional path to save processed images.")
    parser.add_argument("--method", "-m", type=str, default=None, help="Optional method to evaluate.")
    parser.add_argument("--weights", "-w", type=str, default=None, help="Path to model weights (if applicable).")

    args = parser.parse_args()
    inference_fn = get_inference_fn(args.method, args.weights)

    evaluate_dataset(
        gt_dir=Path(args.gt_dir),
        input_dir=Path(args.input_dir),
        results_csv=Path(args.results_csv),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        inference_fn=inference_fn,
    )


if __name__ == "__main__":
    main()
