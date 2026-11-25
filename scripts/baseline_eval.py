"""Baseline: COCO-pretrained YOLOv8n evaluated zero-shot on the ExDark split.

ExDark's 12 classes are a subset of COCO, so the pretrained model can be
evaluated without any fine-tuning. The dataset is rebuilt in YOLO format using
COCO class ids (only the 12 relevant ids carry labels), so metrics are averaged
over the 12 ExDark classes.

Usage (from the project root):
    python scripts/baseline_eval.py [--split test|validation] [--device cpu]
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parent.parent
ANNO_DIR = ROOT / "data/raw/annotations/ExDark_Annno"
SPLIT_DIR = ROOT / "data/outputs/splits"
YOLO_DIR = ROOT / "data/yolo_coco"

# ExDark class -> COCO class id
EXDARK_TO_COCO = {
    "Bicycle": 1, "Boat": 8, "Bottle": 39, "Bus": 5, "Car": 2, "Cat": 15,
    "Chair": 56, "Cup": 41, "Dog": 16, "Motorbike": 3, "People": 0, "Table": 60,
}


def build_split(split: str, coco_names: dict) -> Path:
    """Symlink images and write YOLO labels for one split; return the data yaml."""
    out = YOLO_DIR / split
    img_out, lbl_out = out / "images", out / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SPLIT_DIR / f"{split}.csv")
    n_boxes = 0
    for image, cls in zip(df["image"], df["class"]):
        src = (ROOT / image).resolve()
        name = f"{cls}_{src.name}"
        link = img_out / name
        if not link.exists():
            link.symlink_to(src)

        with Image.open(src) as im:
            w, h = im.size
        lines = []
        ann = ANNO_DIR / cls / f"{src.name}.txt"
        for row in ann.read_text().splitlines():
            if row.startswith("%") or not row.strip():
                continue
            parts = row.split()
            if parts[0] not in EXDARK_TO_COCO:
                continue
            x, y, bw, bh = map(float, parts[1:5])
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + bw, w), min(y + bh, h)
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(
                f"{EXDARK_TO_COCO[parts[0]]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            )
        n_boxes += len(lines)
        (lbl_out / f"{Path(name).stem}.txt").write_text("\n".join(lines))
    print(f"[{split}] {len(df)} images, {n_boxes} boxes")

    data_yaml = YOLO_DIR / f"{split}.yaml"
    YAML.save(data_yaml, {
        "path": str(YOLO_DIR), "train": f"{split}/images", "val": f"{split}/images",
        "names": coco_names,
    })
    return data_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--weights", default="yolov8n.pt")
    args = ap.parse_args()

    model = YOLO(args.weights)
    data_yaml = build_split(args.split, model.names)

    out_dir = ROOT / "outputs/baseline_v8n_exdark"
    m = model.val(
        data=str(data_yaml), imgsz=args.imgsz, batch=args.batch, device=args.device,
        conf=0.001, iou=0.6, workers=4, plots=True, verbose=True,
        project=str(out_dir), name=args.split, exist_ok=True,
    )

    per_class = {}
    for i, c in enumerate(m.ap_class_index):
        p, r, ap50, ap = m.class_result(i)
        per_class[model.names[int(c)]] = {
            "precision": p, "recall": r, "mAP50": ap50, "mAP50-95": ap,
        }
    summary = {
        "weights": args.weights, "split": args.split, "imgsz": args.imgsz,
        "precision": m.box.mp, "recall": m.box.mr,
        "mAP50": m.box.map50, "mAP50-95": m.box.map,
        "speed_ms_per_img": m.speed, "per_class": per_class,
    }
    (out_dir / f"{args.split}_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_class"}, indent=2))
    print(pd.DataFrame(per_class).T.round(3).to_string())


if __name__ == "__main__":
    main()
