import torch
import torch.nn as nn
from config.settings import IMG_SIZE


class Discriminator(nn.Module):
    def __init__(self, n_classes=3, img_channels=3):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, IMG_SIZE * IMG_SIZE)

        def block(in_f, out_f, norm=True):
            layers = [nn.Conv2d(in_f, out_f, 4, 2, 1, bias=False)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_f, affine=True))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers

        self.net = nn.Sequential(
            *block(img_channels + 1, 64, norm=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        label_map = self.label_emb(labels).view(labels.size(0), 1, IMG_SIZE, IMG_SIZE)
        return self.net(torch.cat([img, label_map], dim=1)).view(-1)
