import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_metrics_report(all_results, filenames, out_dir="results/report"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "metrics_test.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Image_ID",
                "Input_PSNR",
                "Input_SSIM",
                "Input_LPIPS",
                "BaselineHE_PSNR",
                "BaselineHE_SSIM",
                "BaselineHE_LPIPS",
                "Network_PSNR",
                "Network_SSIM",
                "Network_LPIPS",
            ]
        )
        for fname, res in zip(filenames, all_results, strict=True):
            writer.writerow(
                [
                    fname,
                    f"{res['Input']['PSNR']:.4f}",
                    f"{res['Input']['SSIM']:.4f}",
                    f"{res['Input']['LPIPS']:.4f}",
                    f"{res['Baseline_HE']['PSNR']:.4f}",
                    f"{res['Baseline_HE']['SSIM']:.4f}",
                    f"{res['Baseline_HE']['LPIPS']:.4f}",
                    f"{res['Network']['PSNR']:.4f}",
                    f"{res['Network']['SSIM']:.4f}",
                    f"{res['Network']['LPIPS']:.4f}",
                ]
            )
    print(f"Zapisano szczegółową tabelę CSV: {csv_path}")

    methods = ["Input", "Baseline_HE", "Network"]
    method_labels = [
        "Input\n(-3 EV + Szum)",
        "Baseline\n(Wyrównanie Histogramu)",
        "HVI-CIDNet\n(Nasz model)",
    ]
    colors = ["#6C757D", "#E76F51", "#2A9D8F"]

    metrics_info = [
        ("PSNR", "PSNR [dB] względem GT (wyżej = lepiej)"),
        ("SSIM", "SSIM [0-1] względem GT (wyżej = lepiej)"),
        ("LPIPS", "LPIPS względem GT (niżej = lepiej)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), dpi=300)
    summary_stats = {m: {} for m in methods}

    for idx, (m_key, m_title) in enumerate(metrics_info):
        ax = axes[idx]
        means, stds = [], []

        for method in methods:
            vals = [res[method][m_key] for res in all_results]
            mean_v, std_v = float(np.mean(vals)), float(np.std(vals))
            means.append(mean_v)
            stds.append(std_v)
            summary_stats[method][m_key] = (mean_v, std_v)

        bars = ax.bar(
            method_labels,
            means,
            yerr=stds,
            capsize=6,
            color=colors,
            edgecolor="black",
            linewidth=1.0,
            alpha=0.9,
            width=0.55,
        )

        ax.set_title(m_title, fontsize=11.5, fontweight="bold", pad=12)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        max_height = max(m + s for m, s in zip(means, stds, strict=True))
        ax.set_ylim(0, max_height * 1.25)

        for bar, mean_v, std_v in zip(bars, means, stds, strict=True):
            fmt = f"{mean_v:.2f} ± {std_v:.2f}" if m_key == "PSNR" else f"{mean_v:.3f} ± {std_v:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + std_v + (max_height * 0.03),
                fmt,
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    fig.suptitle(
        f"Porównanie z Ground Truth na zbiorze testowym RAISE ({len(filenames)} obrazów)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    chart_path = out_dir / "metrics_summary.png"
    plt.savefig(chart_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Zapisano wykres podsumowujący metryki: {chart_path}")
