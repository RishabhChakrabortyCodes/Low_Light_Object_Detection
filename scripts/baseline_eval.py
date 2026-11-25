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
from ultralytics import YOLO

from low_light.data import build_coco_split

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--weights", default="yolov8n.pt")
    args = ap.parse_args()

    model = YOLO(args.weights)
    data_yaml = build_coco_split(args.split, model.names)

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
