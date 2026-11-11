import csv
import multiprocessing as mp
import os

import imageio.v2 as imageio
import numpy as np
import rawpy
from tqdm.contrib.concurrent import process_map


def get_ground_truth(raw):
    """
    Generates the properly exposed, full-color sRGB image (Ground Truth).
    half_size=True is used to speed up processing during training/experiments.
    """
    return raw.postprocess(use_camera_wb=True, half_size=True)


def add_physical_noise(linear_img, alpha=0.001, sigma=0.005):
    """
    Simulates physical camera sensor noise.
    alpha: controls Poisson (photon) noise
    sigma: controls Gaussian (electronic read) noise
    """
    noisy_img = np.random.poisson(linear_img / alpha) * alpha
    noisy_img += np.random.normal(0, sigma, linear_img.shape)
    return np.clip(noisy_img, 0.0, 1.0)


def get_linear_image(raw):
    """
    Extracts the linear image from RAW data.
    Disables auto-brightness and gamma correction.
    Returns a float32 image in the range [0.0, 1.0].
    """
    linear_16bit = raw.postprocess(
        use_camera_wb=True,
        gamma=(1, 1),
        no_auto_bright=True,
        output_bps=16,
        half_size=True,
    )
    return linear_16bit.astype(np.float32) / 65535.0


def reduce_exposure(linear_img, ev_shift):
    """
    Simulate exposure reduction.
    A change of -1 EV halves the amount of light: scale = 2 ^ ev_shift.
    """
    scale_factor = 2.0**ev_shift
    return np.clip(linear_img * scale_factor, 0.0, 1.0)


def apply_gamma_correction(img, gamma=2.2):
    """
    Apply sRGB gamma correction: image ^ (1 / gamma).
    """
    return np.power(np.clip(img, 0.0, 1.0), 1.0 / gamma)


def convert_to_8bit(img):
    """
    Convert the float image (range 0.0 to 1.0) to an 8-bit integer (range 0 to 255).
    """
    return (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def get_low_light(raw, ev_shift=-3.0):
    """
    Generates the degraded, low-light input image.
    Students must complete the helper functions above to make this pipeline work.
    """
    # Step 1: Get linear image data
    linear_img = get_linear_image(raw)
    # Step 2: Mathematically reduce the exposure
    dark_linear = reduce_exposure(linear_img, ev_shift)
    # Step 3: Add physical sensor noise
    noisy_dark_linear = add_physical_noise(dark_linear)
    # Step 4: Apply gamma curve for correct display
    dark_srgb = apply_gamma_correction(noisy_dark_linear)
    # Step 5: Convert back to standard 8-bit format
    final_8bit = convert_to_8bit(dark_srgb)
    return final_8bit


def process_single_file(file_path, output_gt_dir, output_dark_dir, ev_shift):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    gt_path = os.path.join(output_gt_dir, f"{base_name}_gt.png")
    dark_path = os.path.join(output_dark_dir, f"{base_name}_input.png")

    if os.path.exists(gt_path) and os.path.exists(dark_path):
        return

    try:
        with rawpy.imread(file_path) as raw:
            img_gt = get_ground_truth(raw)
            img_dark = get_low_light(raw, ev_shift)
        imageio.imsave(gt_path, img_gt)
        imageio.imsave(dark_path, img_dark)
    except Exception as e:
        print(f"Error processing {base_name}: {e}")


def _worker(args):
    idx, file_path, gt_dir, dark_dir, ev_shift = args
    np.random.seed(42 + idx)
    process_single_file(file_path, gt_dir, dark_dir, ev_shift)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    CSV_PATH = "data/raw/RAISE_258.csv"
    NEF_DIR = "data/raw/nef"
    PROCESSED_DIR = "data/processed"

    LIMIT = 100
    TRAIN_COUNT = 80
    EV_SHIFT = -3.0

    for split in ("train", "test"):
        os.makedirs(os.path.join(PROCESSED_DIR, split, "gt"), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DIR, split, "input"), exist_ok=True)

    with open(CSV_PATH, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))[:LIMIT]

    tasks = []
    for i, row in enumerate(rows):
        filename = row["File"]
        if not filename.upper().endswith(".NEF"):
            filename += ".NEF"

        nef_path = os.path.join(NEF_DIR, filename)
        split = "train" if i < TRAIN_COUNT else "test"
        gt_dir = os.path.join(PROCESSED_DIR, split, "gt")
        dark_dir = os.path.join(PROCESSED_DIR, split, "input")
        tasks.append((i, nef_path, gt_dir, dark_dir, EV_SHIFT))

    process_map(_worker, tasks, desc="Processing RAW images", chunksize=1)
