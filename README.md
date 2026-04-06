# GAN-Augmented Breast Ultrasound Classification

Conditional GAN (cGAN) augmentation pipeline for breast ultrasound image classification. Trains a generator to synthesize class-conditioned images, then benchmarks an EfficientNet-B0 classifier trained on real-only data versus real + GAN-augmented data.

---

## Sample Data

| Benign | Malignant |
|--------|-----------|
| ![benign](assets/benign__74_.png) | ![malignant](assets/malignant__13_.png) |

Benign lesions typically appear as well-defined, homogeneous, hypoechoic masses with smooth borders. Malignant lesions tend to show irregular margins, heterogeneous echotexture, and disrupted surrounding tissue architecture.

---

## Project Structure

```
gan_clf/
├── main.py                        ← Entry point
├── config/
│   └── settings.py                ← All hyperparameters
├── data/
│   ├── datasets.py                ← ClassificationDataset, GANDataset
│   ├── transforms.py              ← GAN / classifier transforms
│   └── loaders.py                 ← Train/val DataLoader builder
├── models/
│   ├── generator.py               ← Conditional Generator (cGAN)
│   ├── discriminator.py           ← Conditional Discriminator
│   └── classifier.py              ← EfficientNet-B0 head
├── training/
│   ├── gan_trainer.py             ← Full cGAN training loop
│   └── clf_trainer.py             ← Classifier training loop
├── evaluation/
│   ├── evaluator.py               ← Inference + metrics
│   └── plots.py                   ← Confusion matrix, curves, bar chart
└── utils/
    └── contact_sheet.py           ← Synthetic image preview grid
```

---

## Setup

```bash
pip install torch torchvision scikit-learn matplotlib seaborn tqdm Pillow
```

Dataset layout expected:

```
dataset/
├── benign/       *.png / *.jpg   (mask files named *_mask.* are auto-skipped)
├── malignant/
└── normal/
```

---

## Usage

```bash
python main.py \
  --data_dir  /path/to/Dataset_BUSI_with_GT \
  --output_dir ./output

# Skip GAN re-training and reuse existing synthetic images
python main.py \
  --data_dir  /path/to/Dataset_BUSI_with_GT \
  --output_dir ./output \
  --skip_gan
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | required | Path to dataset root |
| `--output_dir` | `./output` | Where to save all artefacts |
| `--gan_epochs` | `10000` | cGAN training epochs |
| `--clf_epochs` | `20` | Classifier training epochs |
| `--synthetic_per_class` | `200` | Synthetic images generated per class |
| `--skip_gan` | `False` | Reuse existing synthetics, skip Phase 1 |

---

## Configuration

All hyperparameters live in `config/settings.py`:

| Parameter | Value |
|---|---|
| `IMG_SIZE` | 64 |
| `CLF_SIZE` | 224 |
| `LATENT_DIM` | 256 |
| `GAN_EPOCHS` | 10 000 |
| `CLF_EPOCHS` | 20 |
| `BATCH_SIZE` | 16 |
| `LR_G / LR_D` | 2e-4 |
| `LR_CLF` | 1e-4 |
| `SYNTHETIC_PER_CLASS` | 200 |
| `SEED` | 42 |

---

## Results — BUSI Dataset

**Dataset split** · 780 real images · 80 / 20 train-val split · 200 synthetics per class (600 total)

### Overall Accuracy

| Model | Val Accuracy |
|---|---|
| EfficientNet-B0 (Real Only) | **0.8654** |
| EfficientNet-B0 (Real + GAN Synthetic) | **0.8974** |
| GAN Augmentation Impact | ▲ **+0.0321** |

GAN augmentation improves overall validation accuracy by **+3.21 percentage points**.

---

### Per-Class Report — Real Only

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| benign | 0.84 | 0.95 | 0.89 | 78 |
| malignant | 0.84 | 0.82 | 0.83 | 45 |
| normal | 1.00 | 0.73 | 0.84 | 33 |
| **accuracy** | | | **0.87** | 156 |
| macro avg | 0.89 | 0.83 | 0.86 | 156 |
| weighted avg | 0.87 | 0.87 | 0.86 | 156 |

---

### Per-Class Report — Real + GAN Synthetic

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| benign | 0.88 | 0.94 | 0.91 | 78 |
| malignant | 0.91 | 0.87 | 0.89 | 45 |
| normal | 0.93 | 0.85 | 0.89 | 33 |
| **accuracy** | | | **0.90** | 156 |
| macro avg | 0.91 | 0.88 | 0.89 | 156 |
| weighted avg | 0.90 | 0.90 | 0.90 | 156 |

**Key gains from GAN augmentation:**
- Malignant precision: 0.84 → 0.91 (+7 pp) — the most clinically important class
- Malignant F1: 0.83 → 0.89 (+6 pp)
- Normal recall: 0.73 → 0.85 (+12 pp) — large gain from 200 extra normals (was the smallest class at 133 images)
- Macro F1: 0.86 → 0.89 (+3 pp)

---

## Output Artefacts

```
output/
├── generator.pth                        ← Saved Generator weights
├── classifier_real_only.pth             ← Best real-only classifier
├── classifier_real_plus_synthetic.pth   ← Best augmented classifier
├── synthetic_images/                    ← 200 PNGs per class
│   ├── benign/
│   ├── malignant/
│   └── normal/
├── synthetic_contact_sheet.png          ← Visual preview grid
├── gan_samples/                         ← epoch_XXXX.png grids
├── gan_loss.png                         ← G / D loss curves
├── training_curves.png                  ← Val acc + loss comparison
├── accuracy_comparison.png              ← Bar chart
├── cm_Real_Only.png                     ← Confusion matrix
├── cm_Real_Synthetic.png                ← Confusion matrix
└── results.json                         ← All metrics (machine-readable)
```

---

## Architecture

**Generator** · Conditional embedding → ConvTranspose2d stack (512→256→128→64→32→3) · Tanh output  
**Discriminator** · Label map channel-concatenation → Conv2d stack (64→128→256→512→1) · InstanceNorm · Sigmoid  
**Classifier** · EfficientNet-B0 (ImageNet pretrained) · Final FC replaced with Linear(1280, n_classes) · AdamW + CosineAnnealingLR
