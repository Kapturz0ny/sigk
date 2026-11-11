import os

import imageio
import numpy as np
import rawpy


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
    TODO: Simulate exposure reduction.
    Hint: A change of -1 EV halves the amount of light.
    Calculate the scale factor based on 'ev_shift' and multiply the image by it.
    """
    raise NotImplementedError("Implement exposure reduction logic")


def apply_gamma_correction(img, gamma=2.2):
    """
    TODO: Apply sRGB gamma correction.
    Hint: The simplified formula for gamma correction is image ^ (1 / gamma).
    """
    raise NotImplementedError("Implement gamma correction logic")


def convert_to_8bit(img):
    """
    TODO: Convert the float image (range 0.0 to 1.0) to an 8-bit integer (range 0 to 255).
    Ensure the returned array has the dtype np.uint8.
    """
    raise NotImplementedError("Implement 8-bit conversion logic")


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
        print(f"Skipping {base_name} - already processed.")
        return

    print(f"Processing {base_name}...")
    try:
        with rawpy.imread(file_path) as raw:
            img_gt = get_ground_truth(raw)
            img_dark = get_low_light(raw, ev_shift)
        imageio.imsave(gt_path, img_gt)
        imageio.imsave(dark_path, img_dark)
    except Exception as e:
        print(f"Error processing {base_name}: {e}")
