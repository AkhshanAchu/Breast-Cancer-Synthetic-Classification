from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def save_contact_sheet(syn_dir, classes, output_dir, n_per_class=10):
    cell = 96
    label_w = 130
    W = label_w + n_per_class * cell
    H = len(classes) * cell
    sheet = Image.new("RGB", (W, H), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    for row, cls in enumerate(classes):
        cls_dir = Path(syn_dir) / cls
        imgs = sorted(cls_dir.glob("*.png"))[:n_per_class]
        y = row * cell
        draw.text((6, y + cell // 2 - 8), cls, fill=(230, 230, 230), font=font)
        draw.line([(0, y), (W, y)], fill=(60, 60, 60), width=1)
        for col, img_path in enumerate(imgs):
            thumb = Image.open(img_path).convert("RGB").resize((cell, cell))
            sheet.paste(thumb, (label_w + col * cell, y))

    out_path = Path(output_dir) / "synthetic_contact_sheet.png"
    sheet.save(str(out_path))
    tqdm.write(f"  Contact sheet saved → {out_path}")
