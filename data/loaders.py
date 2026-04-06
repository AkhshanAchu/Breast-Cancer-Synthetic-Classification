import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split

from config.settings import BATCH_SIZE, SEED
from data.datasets import ClassificationDataset
from data.transforms import clf_train_transform, clf_val_transform


def build_clf_loaders(data_dir, syn_dir):
    real_ds = ClassificationDataset(data_dir, clf_train_transform())
    val_size = max(int(0.2 * len(real_ds)), len(real_ds))
    train_size = len(real_ds) - val_size

    generator = torch.Generator().manual_seed(SEED)
    real_train_ds, _ = random_split(real_ds, [train_size, val_size], generator=generator)

    val_ds_clean = ClassificationDataset(data_dir, clf_val_transform())
    generator = torch.Generator().manual_seed(SEED)
    _, val_ds_clean = random_split(val_ds_clean, [train_size, val_size], generator=generator)

    syn_ds = ClassificationDataset(syn_dir, clf_train_transform())
    aug_train_ds = ConcatDataset([real_train_ds, syn_ds])

    real_train_loader = DataLoader(real_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    aug_train_loader  = DataLoader(aug_train_ds,  batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader        = DataLoader(val_ds_clean,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    sizes = {
        "real_train": len(real_train_ds),
        "synthetic":  len(syn_ds),
        "aug_train":  len(aug_train_ds),
        "val":        val_size,
        "train_size": train_size,
    }

    return real_train_loader, aug_train_loader, val_loader, sizes
