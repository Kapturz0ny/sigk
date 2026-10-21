import cv2
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def center_crop_to_square(image: np.ndarray) -> np.ndarray:
    """Crops the given image to a square based on its shortest side."""
    height, width = image.shape[:2]
    min_dim = min(height, width)
    
    start_y = (height - min_dim) // 2
    start_x = (width - min_dim) // 2
    
    return image[start_y : start_y + min_dim, start_x : start_x + min_dim]

def process_directory(input_dir: Path, output_dir: Path, target_size: int = 256) -> None:
    """Crops and resizes all images in the input directory, saving them to the output directory."""
    if not input_dir.exists():
        logging.warning(f"Input directory {input_dir} does not exist. Skipping.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    
    if not image_paths:
        logging.warning(f"No images found in {input_dir}.")
        return

    logging.info(f"Processing {len(image_paths)} images from {input_dir} to {output_dir}...")
    
    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        
        if image is None:
            logging.error(f"Failed to read image: {img_path}")
            continue
            
        cropped_image = center_crop_to_square(image)
        
        resized_image = cv2.resize(
            cropped_image, 
            (target_size, target_size), 
            interpolation=cv2.INTER_AREA
        )
        
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), resized_image)

def main() -> None:
    """Main function to execute the preprocessing pipeline."""
    base_data_dir = Path("data/raw/")
    
    directories_to_process = [
        {
            "input": base_data_dir / "DIV2K_train_HR",
            "output": base_data_dir / "train_256"
        },
        {
            "input": base_data_dir / "DIV2K_valid_HR",
            "output": base_data_dir / "valid_256"
        }
    ]
    
    for dirs in directories_to_process:
        process_directory(input_dir=dirs["input"], output_dir=dirs["output"])
        
    logging.info("Data preprocessing completed successfully.")

if __name__ == "__main__":
    main()