import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_classifier(model, loader, classes, device):
    model.eval()
    all_preds, all_labels_out = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  Evaluating", ncols=80, leave=False):
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels_out.extend(labels.numpy())
    report = classification_report(all_labels_out, all_preds,
                                   target_names=classes, output_dict=True)
    cm = confusion_matrix(all_labels_out, all_preds)
    return report, cm, all_labels_out, all_preds
