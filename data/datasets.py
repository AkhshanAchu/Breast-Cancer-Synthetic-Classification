from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


class ClassificationDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            for f in (self.root / cls).iterdir():
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    if "_mask" not in f.stem:
                        self.samples.append((str(f), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class GANDataset(Dataset):
    def __init__(self, root, class_name, transform=None):
        self.root = Path(root) / class_name
        self.transform = transform
        self.samples = [
            str(f) for f in self.root.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
            and "_mask" not in f.stem
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img,


class CombinedGANDataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.labels[i]
