import argparse
import logging
import cv2
import numpy as np
from pathlib import Path
from skimage.util import random_noise

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def apply_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """Applies Gaussian noise to an image given a sigma value."""
    image_normalized = image.astype(np.float32) / 255.0
    noisy_image = random_noise(image_normalized, mode="gaussian", var=sigma**2, clip=True)
    
    return (noisy_image * 255).astype(np.uint8)


def generate_noisy_dataset(input_dir: Path, output_dir: Path, sigma: float) -> None:
    """Generates a dataset of noisy images for a specific sigma value."""
    if not input_dir.exists():
        logging.error(f"Input directory {input_dir} not found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))

    logging.info(f"Adding noise (sigma={sigma}) to {len(image_paths)} images. Target: {output_dir}")

    for img_path in image_paths:
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        noisy_rgb = apply_gaussian_noise(image_rgb, sigma)
        noisy_bgr = cv2.cvtColor(noisy_rgb, cv2.COLOR_RGB2BGR)
        
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), noisy_bgr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate noisy images for validation/testing.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to clean images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save noisy images.")
    parser.add_argument("--sigma", type=float, required=True, help="Gaussian noise sigma value.")
    
    args = parser.parse_args()
    
    generate_noisy_dataset(Path(args.input_dir), Path(args.output_dir), args.sigma)
    logging.info("Dataset preparation completed.")


if __name__ == "__main__":
    main()
