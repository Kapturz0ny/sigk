import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


def create_zoom_and_hist_comparison(
    inp_np,
    base_np,
    pred_np,
    gt_np,
    metrics_dict,
    save_path,
    roi_1=None,
    roi_2=None,
    title_prefix="",
):
    h, w, _ = gt_np.shape
    crop_size = min(h, w) // 5

    if roi_1 is None:
        roi_1 = (int(w * 0.25), int(h * 0.30), crop_size, crop_size)
    if roi_2 is None:
        roi_2 = (int(w * 0.60), int(h * 0.55), crop_size, crop_size)

    images = [inp_np, base_np, pred_np, gt_np]
    col_names = [
        "Input (-3 EV + Szum)",
        "Baseline (Wyrównanie Histogramu)",
        "HVI-CIDNet (Nasz model)",
        "Ground Truth (Referencja)",
    ]
    metric_keys = ["Input", "Baseline_HE", "Network", None]

    fig = plt.figure(figsize=(18, 13.5), dpi=300)
    gs = fig.add_gridspec(4, 4, height_ratios=[1.35, 0.95, 0.95, 0.75], hspace=0.22, wspace=0.10)

    for col_idx in range(4):
        img = images[col_idx]
        m_key = metric_keys[col_idx]

        ax_main = fig.add_subplot(gs[0, col_idx])
        ax_main.imshow(img)
        ax_main.axis("off")

        if m_key is not None:
            m = metrics_dict[m_key]
            subtitle = f"\nPSNR: {m['PSNR']:.2f} dB | SSIM: {m['SSIM']:.3f}\nLPIPS: {m['LPIPS']:.3f}"
        else:
            subtitle = "\nObraz referencyjny (Ground Truth)"

        ax_main.set_title(f"{col_names[col_idx]}{subtitle}", fontsize=10.5, fontweight="bold", pad=8)

        for roi, color in [(roi_1, "#FF2E2E"), (roi_2, "#00B4D8")]:
            rx, ry, rw, rh = roi
            rect = patches.Rectangle((rx, ry), rw, rh, linewidth=2.2, edgecolor=color, facecolor="none")
            ax_main.add_patch(rect)

        rx1, ry1, rw1, rh1 = roi_1
        crop_1 = img[ry1 : ry1 + rh1, rx1 : rx1 + rw1]
        ax_z1 = fig.add_subplot(gs[1, col_idx])
        ax_z1.imshow(crop_1, interpolation="nearest")
        for spine in ax_z1.spines.values():
            spine.set_edgecolor("#FF2E2E")
            spine.set_linewidth(2.5)
        ax_z1.set_xticks([])
        ax_z1.set_yticks([])
        if col_idx == 0:
            ax_z1.set_ylabel("Detal 1 (Zoom)", fontsize=10.5, fontweight="bold", color="#FF2E2E")

        rx2, ry2, rw2, rh2 = roi_2
        crop_2 = img[ry2 : ry2 + rh2, rx2 : rx2 + rw2]
        ax_z2 = fig.add_subplot(gs[2, col_idx])
        ax_z2.imshow(crop_2, interpolation="nearest")
        for spine in ax_z2.spines.values():
            spine.set_edgecolor("#00B4D8")
            spine.set_linewidth(2.5)
        ax_z2.set_xticks([])
        ax_z2.set_yticks([])
        if col_idx == 0:
            ax_z2.set_ylabel("Detal 2 (Zoom)", fontsize=10.5, fontweight="bold", color="#00B4D8")

        ax_hist = fig.add_subplot(gs[3, col_idx])
        for ch_idx, ch_color in enumerate(["#E63946", "#2A9D8F", "#457B9D"]):
            ch_vals = img[:, :, ch_idx].flatten().astype(np.float32)
            if ch_vals.max() <= 1.0:
                ch_vals *= 255.0
            hist, bins = np.histogram(ch_vals, bins=64, range=(0, 256), density=True)
            ax_hist.plot(bins[:-1], hist, color=ch_color, linewidth=1.2, alpha=0.85)
            ax_hist.fill_between(bins[:-1], hist, color=ch_color, alpha=0.15)

        ax_hist.set_xlim(0, 255)
        ax_hist.set_ylim(0, 0.035)
        ax_hist.grid(True, linestyle="--", alpha=0.4)
        ax_hist.tick_params(axis="both", labelsize=8)
        ax_hist.set_xlabel("Jasność pikseli [0-255]", fontsize=8.5)
        if col_idx == 0:
            ax_hist.set_ylabel("Gęstość (Histogram)", fontsize=9.5, fontweight="bold")

    if title_prefix:
        fig.suptitle(
            f"Analiza wizualna i histogramowa: {title_prefix}",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Zapisano planszę porównawczą: {save_path}")
