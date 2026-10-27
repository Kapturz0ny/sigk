import random
from pathlib import Path

import cv2
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, v2

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def list_images(directory: Path) -> list[Path]:
    """Returns a sorted list of supported image files in a directory."""
    return sorted(
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )


class SRDataset(Dataset):
    """Dataset returning paired low-resolution and high-resolution image tensors."""

    def __init__(
        self,
        image_paths: list[Path],
        hr_size: int = 256,
        scale: int = 4,
        augment: bool = False,
    ) -> None:
        if hr_size % scale != 0:
            raise ValueError(
                f"hr_size ({hr_size}) must be divisible by scale ({scale})."
            )

        self.image_paths = image_paths
        self.hr_size = hr_size
        self.lr_size = hr_size // scale
        self.augment = (
            v2.Compose([v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5)])
            if augment
            else None
        )
        self.to_float = v2.ToDtype(torch.float32, scale=True)
        self.hr_resize = v2.Resize(
            (hr_size, hr_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.lr_resize = v2.Resize(
            (self.lr_size, self.lr_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def _load_image(self, path: Path) -> torch.Tensor:
        """Reads an RGB image and returns it as a uint8 CHW tensor."""
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(img_rgb).permute(2, 0, 1)

        if image.shape[-2:] != (self.hr_size, self.hr_size):
            image = self.hr_resize(image)

        return image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a synchronized pair of (lr_tensor, hr_tensor), both in [0, 1]."""
        hr_image = self._load_image(self.image_paths[idx])

        if self.augment is not None:
            hr_image = self.augment(hr_image)

        hr_image = self.to_float(hr_image)
        lr_image = self.lr_resize(hr_image)

        return lr_image, hr_image


class SRDataModule(L.LightningDataModule):
    """LightningDataModule managing DIV2K train/validation/test splits and DataLoaders."""

    def __init__(
        self,
        train_dir: Path,
        test_dir: Path,
        batch_size: int = 16,
        num_workers: int = 4,
        hr_size: int = 256,
        scale: int = 4,
        val_split: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.hr_size = hr_size
        self.scale = scale
        self.val_split = val_split
        self.seed = seed
        self.train_dataset: SRDataset | None = None
        self.val_dataset: SRDataset | None = None
        self.train_paths: list[Path] = []
        self.val_paths: list[Path] = []
        self.test_paths: list[Path] = []

    def setup(self, stage: str | None = None) -> None:
        """Splits the DIV2K validation folder into validation and test subsets."""
        valid_paths = list_images(self.train_dir)
        self.test_paths = list_images(self.test_dir)

        if not self.test_paths:
            raise FileNotFoundError(f"No images found in {self.test_paths}")
        if len(valid_paths) < 2:
            raise FileNotFoundError(
                f"At least two images are required in {valid_paths} to create "
                "validation and train splits."
            )

        random.Random(self.seed).shuffle(valid_paths)
        val_count = min(
            max(1, int(len(valid_paths) * self.val_split)), len(valid_paths) - 1
        )

        self.val_paths = valid_paths[:val_count]
        self.train_paths = valid_paths[val_count:]

        self.train_dataset = SRDataset(
            self.train_paths, hr_size=self.hr_size, scale=self.scale, augment=True
        )
        self.val_dataset = SRDataset(
            self.val_paths, hr_size=self.hr_size, scale=self.scale, augment=False
        )
        self.test_dataset = SRDataset(
            self.test_paths, hr_size=self.hr_size, scale=self.scale, augment=False
        )

    def train_dataloader(self) -> DataLoader:
        """Returns the training DataLoader with persistent workers and a seeded generator."""
        assert self.train_dataset is not None
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
            generator=generator,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation DataLoader."""
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the validation DataLoader."""
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
