import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config.settings import LR_CLF, CLF_EPOCHS
from models.classifier import build_classifier


def train_classifier(train_loader, val_loader, n_classes, device,
                     n_epochs=CLF_EPOCHS, tag="model", output_dir="."):
    model = build_classifier(n_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR_CLF, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'─' * 60}")
    print(f"  Classifier: {tag}")
    print(f"{'─' * 60}")

    epoch_bar = tqdm(range(1, n_epochs + 1), desc=f"  [{tag[:20]}]", ncols=90, unit="epoch")

    for epoch in epoch_bar:
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * imgs.size(0)
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total += imgs.size(0)
        scheduler.step()

        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                v_loss += loss.item() * imgs.size(0)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total += imgs.size(0)

        t_acc = t_correct / t_total
        v_acc = v_correct / v_total
        history["train_loss"].append(t_loss / t_total)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss / v_total)
        history["val_acc"].append(v_acc)

        if v_acc > best_acc:
            best_acc = v_acc
            best_wts = copy.deepcopy(model.state_dict())

        epoch_bar.set_postfix(train=f"{t_acc:.3f}", val=f"{v_acc:.3f}", best=f"{best_acc:.3f}")

    model.load_state_dict(best_wts)
    torch.save(model.state_dict(), str(Path(output_dir) / f"classifier_{tag}.pth"))
    tqdm.write(f"  ✓ Best Val Acc [{tag}]: {best_acc:.4f}")
    return model, history, best_acc
