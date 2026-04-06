import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report

from config.settings import GAN_EPOCHS, CLF_EPOCHS, SYNTHETIC_PER_CLASS, SEED
from data.loaders import build_clf_loaders
from training.gan_trainer import train_cgan
from training.clf_trainer import train_classifier
from evaluation.evaluator import evaluate_classifier
from evaluation.plots import (
    plot_confusion_matrix,
    plot_training_curves,
    plot_accuracy_bar,
)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",            type=str, required=True)
    parser.add_argument("--output_dir",          type=str, default="./output")
    parser.add_argument("--gan_epochs",          type=int, default=GAN_EPOCHS)
    parser.add_argument("--clf_epochs",          type=int, default=CLF_EPOCHS)
    parser.add_argument("--synthetic_per_class", type=int, default=SYNTHETIC_PER_CLASS)
    parser.add_argument("--skip_gan",            action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = Path(args.data_dir)
    classes = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    print(f"\nDetected {len(classes)} classes: {classes}")
    assert len(classes) >= 2, "Need at least 2 classes"

    syn_dir = str(Path(args.output_dir) / "synthetic_images")
    if not args.skip_gan:
        syn_dir, _, _ = train_cgan(
            args.data_dir, classes, args.output_dir, device,
            n_epochs=args.gan_epochs,
            synthetic_per_class=args.synthetic_per_class,
        )
    else:
        print("\nSkipping GAN training — using existing synthetic images.")

    print("\n" + "=" * 60)
    print("  PHASE 2: Training Classifiers")
    print("=" * 60)

    real_train_loader, aug_train_loader, val_loader, sizes = build_clf_loaders(
        args.data_dir, syn_dir
    )

    print(f"\nTotal real images: {sizes['real_train'] + sizes['val']}")
    print(f"  Real-only train : {sizes['real_train']}")
    print(f"  Synthetic images: {sizes['synthetic']}")
    print(f"  Augmented train : {sizes['aug_train']}")
    print(f"  Validation      : {sizes['val']}")

    n_classes = len(classes)

    model_real, hist_real, acc_real = train_classifier(
        real_train_loader, val_loader, n_classes, device,
        n_epochs=args.clf_epochs, tag="real_only", output_dir=args.output_dir,
    )

    model_aug, hist_aug, acc_aug = train_classifier(
        aug_train_loader, val_loader, n_classes, device,
        n_epochs=args.clf_epochs, tag="real_plus_synthetic", output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("  PHASE 3: Evaluation")
    print("=" * 60)

    report_real, cm_real, true_real, pred_real = evaluate_classifier(
        model_real, val_loader, classes, device
    )
    report_aug, cm_aug, true_aug, pred_aug = evaluate_classifier(
        model_aug, val_loader, classes, device
    )

    plot_confusion_matrix(cm_real, classes, "Real_Only",      args.output_dir)
    plot_confusion_matrix(cm_aug,  classes, "Real_Synthetic", args.output_dir)
    plot_training_curves(hist_real, hist_aug, args.output_dir)
    plot_accuracy_bar(acc_real, acc_aug, args.output_dir)

    delta = acc_aug - acc_real
    arrow = "▲" if delta >= 0 else "▼"

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  {'Model':<38} {'Val Accuracy':>10}")
    print(f"  {'─' * 50}")
    print(f"  {'EfficientNet-B0  (Real Only)':<38} {acc_real:>10.4f}")
    print(f"  {'EfficientNet-B0  (Real + GAN Synthetic)':<38} {acc_aug:>10.4f}")
    print(f"\n  GAN Augmentation Impact: {arrow} {abs(delta):.4f}  "
          f"({'improvement' if delta >= 0 else 'degradation'})")
    print("\n  Per-class report — Real Only:")
    print(classification_report(true_real, pred_real, target_names=classes))
    print("  Per-class report — Real + Synthetic:")
    print(classification_report(true_aug, pred_aug, target_names=classes))

    results = {
        "classes": classes,
        "real_train_size": sizes["train_size"],
        "synthetic_per_class": args.synthetic_per_class,
        "val_size": sizes["val"],
        "accuracy_real_only": acc_real,
        "accuracy_real_plus_synthetic": acc_aug,
        "delta_accuracy": delta,
        "per_class_real": report_real,
        "per_class_aug": report_aug,
    }
    with open(str(Path(args.output_dir) / "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  All outputs in: {args.output_dir}/")
    print("  ├── generator.pth")
    print("  ├── classifier_real_only.pth")
    print("  ├── classifier_real_plus_synthetic.pth")
    print("  ├── synthetic_images/")
    print("  ├── synthetic_contact_sheet.png")
    print("  ├── gan_samples/")
    print("  ├── gan_loss.png")
    print("  ├── training_curves.png")
    print("  ├── accuracy_comparison.png")
    print("  ├── cm_Real_Only.png")
    print("  ├── cm_Real_Synthetic.png")
    print("  └── results.json")


if __name__ == "__main__":
    main()
