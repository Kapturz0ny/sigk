"""Generowanie plansz porównawczych i raportu metryk na zbiorze testowym."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
from tqdm import tqdm

from src.metrics import MetricsEvaluator
from src.visualization import create_zoom_and_hist_comparison, save_metrics_report

TEST_DIR = Path("data/processed/test")
GT_DIR = TEST_DIR / "gt"
INPUT_DIR = TEST_DIR / "input"
BASELINE_DIR = TEST_DIR / "baseline"
NETWORK_DIR = TEST_DIR / "hvi_cidnet"
DEFAULT_OUT_DIR = Path("results/report")


def collect_test_results(limit: int | None = None) -> list[dict[str, Path | str]]:
    """Zbiera pary obrazów wejściowych, GT, baseline i predykcji sieci."""
    input_paths = sorted(INPUT_DIR.glob("*_input.png"))
    if not input_paths:
        raise FileNotFoundError(f"Brak obrazów wejściowych w {INPUT_DIR}. Uruchom najpierw src/data/process.py.")

    if limit is not None:
        input_paths = input_paths[:limit]

    results = []
    missing = []
    for input_path in input_paths:
        base_name = input_path.name.removesuffix("_input.png")
        gt_path = GT_DIR / f"{base_name}_gt.png"
        baseline_path = BASELINE_DIR / f"{base_name}.png"
        network_path = NETWORK_DIR / f"{base_name}.png"

        for path in (gt_path, baseline_path, network_path):
            if not path.exists():
                missing.append(path)

        results.append(
            {
                "name": base_name,
                "input": input_path,
                "gt": gt_path,
                "baseline": baseline_path,
                "network": network_path,
            }
        )

    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing[:10])
        raise FileNotFoundError(
            "Brakuje plików wynikowych potrzebnych do wizualizacji:\n"
            f"{missing_list}\n"
            "Uruchom najpierw ewaluację baseline i sieci, np.:\n"
            "  python -m src.evaluate --method baseline\n"
            "  python -m src.evaluate --method hvi_cidnet --weights models/hvi_cidnet/last.ckpt"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generuje plansze porównawcze i raport metryk dla zbioru testowego.")
    parser.add_argument(
        "--max-plots",
        type=int,
        default=5,
        help="Liczba obrazów, dla których zapisywane są plansze 4x4 (domyślnie 5).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maksymalna liczba obrazów testowych branych do raportu (domyślnie wszystkie).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Katalog wynikowy (domyślnie {DEFAULT_OUT_DIR}).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = collect_test_results(limit=args.limit)
    evaluator = MetricsEvaluator()

    all_metrics = []
    image_ids = []

    for idx, result in enumerate(tqdm(results, desc="Generowanie wizualizacji i metryk")):
        base_name = str(result["name"])
        image_ids.append(base_name)

        input_img = imageio.imread(result["input"])
        gt_img = imageio.imread(result["gt"])
        baseline_img = imageio.imread(result["baseline"])
        network_img = imageio.imread(result["network"])

        metrics = {
            "Input": evaluator.evaluate(gt_img, input_img),
            "Baseline_HE": evaluator.evaluate(gt_img, baseline_img),
            "Network": evaluator.evaluate(gt_img, network_img),
        }
        all_metrics.append(metrics)

        if idx < args.max_plots:
            create_zoom_and_hist_comparison(
                input_img,
                baseline_img,
                network_img,
                gt_img,
                metrics_dict=metrics,
                save_path=args.out_dir / f"comparison_{base_name}.png",
                title_prefix=base_name,
            )

    save_metrics_report(all_metrics, image_ids, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
