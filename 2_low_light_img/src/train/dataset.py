import random
from pathlib import Path

import imageio.v2 as imageio
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


class PairedLowLightDataset(Dataset):
    """Dataset loading paired low-light and ground-truth images with random patch cropping."""

    def __init__(
        self,
        input_paths: list[Path],
        gt_dir: Path,
        patch_size: int | None = 384,
        repeat_factor: int = 1,
        augment: bool = False,
    ) -> None:
        self.input_paths = input_paths
        self.gt_dir = gt_dir
        self.patch_size = patch_size
        self.repeat_factor = repeat_factor if augment else 1
        self.augment = augment

        self.flip_transform = (
            v2.Compose([v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5)]) if augment else None
        )

        # Cache decoded uint8 images in RAM (80 images ~ 950 MB) for fast epoch iteration
        self._cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for in_path in self.input_paths:
            gt_name = in_path.name.replace("_input.png", "_gt.png")
            gt_path = self.gt_dir / gt_name
            in_img = torch.from_numpy(imageio.imread(in_path)).permute(2, 0, 1)
            gt_img = torch.from_numpy(imageio.imread(gt_path)).permute(2, 0, 1)
            self._cache.append((in_img, gt_img))

    def __len__(self) -> int:
        return len(self.input_paths) * self.repeat_factor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = idx % len(self.input_paths)
        in_uint8, gt_uint8 = self._cache[real_idx]

        _, h, w = in_uint8.shape
        if self.patch_size is not None and h >= self.patch_size and w >= self.patch_size:
            if self.augment:
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
            else:
                top = (h - self.patch_size) // 2
                left = (w - self.patch_size) // 2

            in_uint8 = in_uint8[:, top : top + self.patch_size, left : left + self.patch_size]
            gt_uint8 = gt_uint8[:, top : top + self.patch_size, left : left + self.patch_size]

        input_tensor = in_uint8.float().div(255.0)
        gt_tensor = gt_uint8.float().div(255.0)

        if self.flip_transform is not None:
            stacked = self.flip_transform(torch.stack([input_tensor, gt_tensor]))
            if random.random() < 0.5:
                stacked = stacked.transpose(-2, -1)
            input_tensor, gt_tensor = stacked[0], stacked[1]

        return input_tensor, gt_tensor


class LowLightDataModule(L.LightningDataModule):
    """LightningDataModule managing train/validation splits and DataLoaders."""

    def __init__(
        self,
        train_dir: Path,
        batch_size: int = 4,
        patch_size: int = 384,
        repeat_factor: int = 10,
        val_split: float = 0.1,
        num_workers: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.input_dir = train_dir / "input"
        self.gt_dir = train_dir / "gt"
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.repeat_factor = repeat_factor
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed

        self.train_dataset: PairedLowLightDataset | None = None
        self.val_dataset: PairedLowLightDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        all_paths = sorted(self.input_dir.glob("*_input.png"))
        if not all_paths:
            raise FileNotFoundError(f"No input images found in {self.input_dir}")

        random.Random(self.seed).shuffle(all_paths)
        val_count = max(1, int(len(all_paths) * self.val_split))

        self.val_dataset = PairedLowLightDataset(
            all_paths[:val_count],
            self.gt_dir,
            patch_size=None,
            repeat_factor=1,
            augment=False,
        )
        self.train_dataset = PairedLowLightDataset(
            all_paths[val_count:],
            self.gt_dir,
            patch_size=self.patch_size,
            repeat_factor=self.repeat_factor,
            augment=True,
        )

    def train_dataloader(self) -> DataLoader:
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
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
