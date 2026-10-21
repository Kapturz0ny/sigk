import random
import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import lightning as L


class PairedImageDataset(Dataset):
    """Dataset loading paired images with augmentations."""

    def __init__(self, input_paths: list[Path], gt_dir: Path, augment: bool = False) -> None:
        self.input_paths = input_paths
        self.gt_dir = gt_dir
        self.transform = (
            v2.Compose([v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5)])
            if augment
            else None
        )

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.input_paths)

    def _load_tensor(self, path: Path) -> torch.Tensor:
        """Reads an image from disk and converts it to a normalized RGB float tensor."""
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img_rgb).float().permute(2, 0, 1).div(255.0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a synchronized pair of (input_tensor, gt_tensor)."""
        input_path = self.input_paths[idx]
        input_tensor = self._load_tensor(input_path)
        gt_tensor = self._load_tensor(self.gt_dir / input_path.name)

        if self.transform is not None:
            stacked = self.transform(torch.stack([input_tensor, gt_tensor]))
            input_tensor, gt_tensor = stacked[0], stacked[1]

        return input_tensor, gt_tensor


class DenoiseDataModule(L.LightningDataModule):
    """LightningDataModule managing reproducible train/validation splits and DataLoaders."""

    def __init__(
        self,
        input_dir: Path,
        gt_dir: Path,
        batch_size: int = 16,
        val_split: float = 0.05,
        num_workers: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        self.train_dataset: PairedImageDataset | None = None
        self.val_dataset: PairedImageDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Splits input images into training and internal validation subsets using the seed."""
        all_paths = sorted(list(self.input_dir.glob("*.png")) + list(self.input_dir.glob("*.jpg")))
        if not all_paths:
            raise FileNotFoundError(f"No images found in {self.input_dir}")

        random.Random(self.seed).shuffle(all_paths)
        val_count = max(1, int(len(all_paths) * self.val_split))

        self.val_dataset = PairedImageDataset(all_paths[:val_count], self.gt_dir, augment=False)
        self.train_dataset = PairedImageDataset(all_paths[val_count:], self.gt_dir, augment=True)

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
        """Returns the validation DataLoader with persistent workers."""
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
