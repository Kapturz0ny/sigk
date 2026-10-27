import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

METRIC_NAMES = ("PSNR", "SSIM", "SNR", "LPIPS")
METHODS = ("opencv_bicubic", "nafnet_sr")


def save_metric_plot(summaries: pd.DataFrame, output_path: Path) -> None:
    """Saves grouped bar charts comparing both methods for every metric and scale."""
    scales = sorted(summaries["scale"].unique())
    positions = np.arange(len(scales))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, metric in zip(axes.flat, METRIC_NAMES):
        for offset, method in zip((-0.5, 0.5), METHODS):
            values = []
            errors = []
            for scale in scales:
                row = summaries[
                    (summaries["scale"] == scale) & (summaries["method"] == method)
                ]
                values.append(
                    row[f"{metric}_mean"].iloc[0] if not row.empty else np.nan
                )
                errors.append(row[f"{metric}_std"].iloc[0] if not row.empty else 0.0)

            axis.bar(
                positions + offset * width,
                values,
                width,
                yerr=errors,
                capsize=4,
                label=method,
            )

        axis.set_xticks(positions)
        axis.set_xticklabels([f"x{int(scale)}" for scale in scales])
        axis.set_title(metric)
        axis.grid(axis="y", alpha=0.3)
        axis.legend()

    fig.suptitle("Super-resolution: NAFNet vs OpenCV bicubic")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def collect_results(
    results_dir: Path,
    output_dir: Path,
    make_plot: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combines per-scale evaluation summaries into final comparison tables."""
    summary_files = sorted(results_dir.glob("*/summary.csv"))
    comparison_files = sorted(results_dir.glob("*/comparison.csv"))

    if not summary_files:
        raise FileNotFoundError(
            f"No summary.csv files found under {results_dir}. Run evaluate.py first."
        )

    summaries = pd.concat(
        [pd.read_csv(path) for path in summary_files],
        ignore_index=True,
    )
    summaries["method"] = pd.Categorical(
        summaries["method"], categories=METHODS, ordered=True
    )
    summaries = summaries.sort_values(["scale", "method"])

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(output_dir / "summary_all_scales.csv", index=False)

    if comparison_files:
        comparisons = pd.concat(
            [pd.read_csv(path) for path in comparison_files],
            ignore_index=True,
        )
        comparisons["metric"] = pd.Categorical(
            comparisons["metric"], categories=METRIC_NAMES, ordered=True
        )
        comparisons = comparisons.sort_values(["scale", "metric"])
        comparisons.to_csv(output_dir / "comparison_all_scales.csv", index=False)
    else:
        comparisons = pd.DataFrame()

    if make_plot:
        plot_path = output_dir / "comparison_all_scales.png"
        save_metric_plot(summaries, plot_path)
        logger.info(f"Saved comparison plot to {plot_path}")

    logger.info(f"Saved combined summary to {output_dir / 'summary_all_scales.csv'}")
    if not comparisons.empty:
        logger.info(
            f"Saved combined comparison to {output_dir / 'comparison_all_scales.csv'}"
        )

    return summaries, comparisons


def main() -> None:
    """Parses arguments and collects all scale-level evaluation results."""
    parser = argparse.ArgumentParser(
        description="Collect and compare super-resolution evaluation results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/upscale",
        help="Directory containing per-scale result subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for combined tables and the plot (default: results_dir).",
    )
    parser.add_argument(
        "--no_plot", action="store_true", help="Skip generating the bar chart."
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    summaries, comparisons = collect_results(
        results_dir=results_dir,
        output_dir=output_dir,
        make_plot=not args.no_plot,
    )

    print("\nCombined summary:")
    print(summaries.to_string(index=False))
    if not comparisons.empty:
        print("\nCombined comparison with OpenCV bicubic:")
        print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
