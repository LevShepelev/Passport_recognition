from pathlib import Path
from typing import Tuple

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class PassportDataset(Dataset):
    """Dataset for passport images and corresponding country labels."""

    def __init__(self, root: str, transform: T.Compose | None = None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        for country_dir in self.root.iterdir():
            if not country_dir.is_dir():
                continue
            label = country_dir.name
            for img_path in country_dir.glob("*.jpg"):
                self.samples.append((img_path, label))

        self.label_to_idx = {
            label: idx for idx, label in enumerate(sorted({s[1] for s in self.samples}))
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label_to_idx[label]
