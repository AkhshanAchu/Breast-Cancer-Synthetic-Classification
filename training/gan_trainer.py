import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import LATENT_DIM, BATCH_SIZE, LR_G, LR_D, GAN_EPOCHS, SYNTHETIC_PER_CLASS
from data.datasets import GANDataset, CombinedGANDataset
from data.transforms import gan_transform
from models.generator import Generator
from models.discriminator import Discriminator
from utils.contact_sheet import save_contact_sheet


def _cache_images(data_dir, classes):
    all_imgs, all_labels = [], []
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print("\nLoading real images:")
    for cls in classes:
        ds = GANDataset(data_dir, cls, gan_transform())
        print(f"  [{cls}]  {len(ds)} images")
        for (img,) in tqdm(ds, desc=f"    Caching {cls}", leave=False, ncols=80):
            all_imgs.append(img)
            all_labels.append(class_to_idx[cls])
    return all_imgs, all_labels


def _save_gan_loss_plot(g_losses, d_losses, output_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(g_losses, label="Generator")
    plt.plot(d_losses, label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / "gan_loss.png"))
    plt.close()


def _generate_synthetic(G, classes, output_dir, device, synthetic_per_class):
    syn_dir = Path(output_dir) / "synthetic_images"
    print(f"\nGenerating {synthetic_per_class} synthetic images per class...")
    G.eval()
    with torch.no_grad():
        for cls_idx, cls_name in enumerate(classes):
            cls_syn_dir = syn_dir / cls_name
            cls_syn_dir.mkdir(parents=True, exist_ok=True)
            generated = 0
            pbar = tqdm(total=synthetic_per_class,
                        desc=f"  Generating [{cls_name}]", unit="img", ncols=80)
            while generated < synthetic_per_class:
                batch = min(BATCH_SIZE, synthetic_per_class - generated)
                z = torch.randn(batch, LATENT_DIM, 1, 1).to(device)
                lbs = torch.full((batch,), cls_idx, dtype=torch.long).to(device)
                imgs = G(z, lbs) * 0.5 + 0.5
                for i, img_t in enumerate(imgs):
                    save_image(img_t, str(cls_syn_dir / f"syn_{generated+i:05d}.png"))
                generated += batch
                pbar.update(batch)
            pbar.close()
    return str(syn_dir)


def train_cgan(data_dir, classes, output_dir, device,
               n_epochs=GAN_EPOCHS, synthetic_per_class=SYNTHETIC_PER_CLASS):
    print("\n" + "=" * 60)
    print("  PHASE 1: Training Conditional GAN")
    print("=" * 60)

    n_classes = len(classes)
    all_imgs, all_labels = _cache_images(data_dir, classes)

    loader = DataLoader(
        CombinedGANDataset(all_imgs, all_labels),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    G = Generator(LATENT_DIM, n_classes).to(device)
    D = Discriminator(n_classes).to(device)
    opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    g_losses, d_losses = [], []
    fixed_z = torch.randn(n_classes * 8, LATENT_DIM, 1, 1).to(device)
    fixed_labels = torch.tensor(list(range(n_classes)) * 8).to(device)
    samples_dir = Path(output_dir) / "gan_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    epoch_bar = tqdm(range(1, n_epochs + 1), desc="  GAN Training", ncols=90, unit="epoch")

    for epoch in epoch_bar:
        g_epoch, d_epoch = [], []

        for real_imgs, labels in loader:
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            bs = real_imgs.size(0)

            real_t = torch.full((bs,), 0.9).to(device)
            fake_t = torch.zeros(bs).to(device)

            z = torch.randn(bs, LATENT_DIM, 1, 1).to(device)
            fake_imgs = G(z, labels).detach()
            loss_D = (criterion(D(real_imgs, labels), real_t) +
                      criterion(D(fake_imgs, labels), fake_t)) / 2
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            z = torch.randn(bs, LATENT_DIM, 1, 1).to(device)
            fake_imgs = G(z, labels)
            loss_G = criterion(D(fake_imgs, labels), torch.ones(bs).to(device))
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            g_epoch.append(loss_G.item())
            d_epoch.append(loss_D.item())

        g_losses.append(np.mean(g_epoch))
        d_losses.append(np.mean(d_epoch))
        epoch_bar.set_postfix(G=f"{g_losses[-1]:.4f}", D=f"{d_losses[-1]:.4f}")

        if epoch % 10 == 0 or epoch == 1:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_z, fixed_labels) * 0.5 + 0.5
            G.train()
            save_image(samples, str(samples_dir / f"epoch_{epoch:04d}.png"), nrow=8)

    _save_gan_loss_plot(g_losses, d_losses, output_dir)

    syn_dir = _generate_synthetic(G, classes, output_dir, device, synthetic_per_class)

    torch.save(G.state_dict(), str(Path(output_dir) / "generator.pth"))
    tqdm.write(f"  Generator saved → {Path(output_dir) / 'generator.pth'}")

    save_contact_sheet(syn_dir, classes, output_dir)

    return syn_dir, g_losses, d_losses
