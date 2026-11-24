import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import PhongSceneDataset


class PhongSceneDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        val_batch_size: int,
        num_workers: int,
        num_fixed_samples: int = 16,
        fixed_seed: int = 1234,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.num_fixed_samples = num_fixed_samples
        self.fixed_seed = fixed_seed
        self.fixed_params = None

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train = PhongSceneDataset(self.data_dir, split="train")
            self.val = PhongSceneDataset(self.data_dir, split="val")
            rng = np.random.default_rng(self.fixed_seed)
            sel = rng.choice(
                len(self.train),
                size=min(self.num_fixed_samples, len(self.train)),
                replace=False,
            )
            self.fixed_params = torch.from_numpy(
                self.train.vector[self.train.indices[sel]]
            ).float()

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
