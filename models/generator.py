import torch
import torch.nn as nn
from config.settings import LATENT_DIM


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, n_classes=3, img_channels=3):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, latent_dim)

        def block(in_f, out_f):
            return [
                nn.ConvTranspose2d(in_f, out_f, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_f),
                nn.ReLU(True),
            ]

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            *block(512, 256),
            *block(256, 128),
            *block(128, 64),
            *block(64, 32),
            nn.Conv2d(32, img_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        emb = self.label_emb(labels).unsqueeze(-1).unsqueeze(-1)
        return self.net(torch.cat([z, emb], dim=1))
