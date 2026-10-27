import argparse
import logging
from pathlib import Path

from src.process_init_data import process_directory

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_div2k_dataset(
    raw_dir: Path, output_dir: Path, target_size: int = 256
) -> None:
    """Crops DIV2K images to squares and resizes them to the target HR size."""
    process_directory(
        input_dir=raw_dir / "DIV2K_train_HR",
        output_dir=output_dir / f"train_{target_size}",
        target_size=target_size,
    )
    process_directory(
        input_dir=raw_dir / "DIV2K_valid_HR",
        output_dir=output_dir / f"valid_{target_size}",
        target_size=target_size,
    )


def main() -> None:
    """Parses arguments and prepares the DIV2K train/validation splits."""
    parser = argparse.ArgumentParser(
        description="Prepare DIV2K HR images for super-resolution training."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw",
        help="Directory containing DIV2K_train_HR and DIV2K_valid_HR.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory for the prepared train_256 and valid_256 splits.",
    )
    parser.add_argument(
        "--target_size", type=int, default=256, help="Target HR image size."
    )

    args = parser.parse_args()
    prepare_div2k_dataset(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        target_size=args.target_size,
    )
    logger.info("DIV2K preparation completed successfully.")


if __name__ == "__main__":
    main()
