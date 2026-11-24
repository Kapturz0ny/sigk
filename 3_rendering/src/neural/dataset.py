import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PhongSceneDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        params = np.load(self.root / "params.npz", allow_pickle=True)
        self.object_position = params["object_position"]
        self.diffuse_color = params["diffuse_color"]
        self.shininess = params["shininess"]
        self.light_position = params["light_position"]
        self.vector = params["vector"]

        with open(self.root / "split.json") as f:
            split_data = json.load(f)
        self.indices = np.asarray(split_data[split])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        img = Image.open(self.root / "images" / f"image_{idx:05}.png")
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2 - 1  # CHW, [-1,1]
        vec = torch.from_numpy(self.vector[idx])
        return tensor, vec, idx
