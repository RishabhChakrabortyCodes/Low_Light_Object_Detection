"""Fine-tune YOLOv8n on ExDark (12 classes) and evaluate it on the held-out split.

Everything is driven by a config (default: configs/baseline.yaml). Pipeline:
  1. build the 12-class YOLO dataset from data/outputs/splits/*.csv
  2. fine-tune from the pretrained weights on train, validating on validation
  3. evaluate best.pt on eval.split and write metrics JSON + plots

Usage (from the project root):
    python scripts/baseline_train.py                       # full run from the config
    python scripts/baseline_train.py --device 0            # pick a GPU
    python scripts/baseline_train.py --epochs 1 --fraction 0.05 --name smoke   # smoke test
    python scripts/baseline_train.py --eval-only runs.pt   # skip training, evaluate weights
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from ultralytics import YOLO
from ultralytics.utils import YAML

from low_light.data import build_exdark_dataset

ROOT = Path(__file__).resolve().parent.parent
SPLIT_ALIASES = {"val": "validation", "validation": "validation", "test": "test"}


def load_config(path: Path) -> dict:
    cfg = YAML.load(path)
    cfg["output"]["dir"] = cfg["output"]["dir"].replace("${run_name}", cfg["run_name"])
    return cfg


def train_args(cfg: dict, data_yaml: Path, out_dir: Path, args) -> dict:
    """Translate the config sections into ultralytics train() arguments."""
    t = cfg["train"]
    kw = {
        "data": str(data_yaml),
        "imgsz": cfg["data"]["imgsz"],
        "seed": cfg["seed"],
        "project": str(out_dir),
        "name": args.name,
        "exist_ok": True,
        "pretrained": True,
        "plots": True,
        **{k: t[k] for k in (
            "epochs", "batch", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
            "warmup_epochs", "patience", "workers", "amp", "cache")},
        **cfg.get("augment", {}),
    }
    if args.epochs is not None:
        kw["epochs"] = args.epochs
    if args.batch is not None:
        kw["batch"] = args.batch
    if args.fraction is not None:
        kw["fraction"] = args.fraction
    if args.device is not None:
        kw["device"] = args.device
    return kw


def evaluate(weights: Path, data_yaml: Path, cfg: dict, split: str, out_dir: Path, args) -> dict:
    model = YOLO(str(weights))
    ev = cfg["eval"]
    m = model.val(
        data=str(data_yaml), split="val" if split == "validation" else "test",
        imgsz=cfg["data"]["imgsz"], batch=args.batch or cfg["train"]["batch"],
        conf=ev["conf"], iou=ev["iou"], device=args.device, workers=cfg["train"]["workers"],
        plots=True, project=str(out_dir), name=f"{args.name}_eval_{split}", exist_ok=True,
    )
    per_class = {}
    for i, c in enumerate(m.ap_class_index):
        p, r, ap50, ap = m.class_result(i)
        per_class[model.names[int(c)]] = {
            "precision": p, "recall": r, "mAP50": ap50, "mAP50-95": ap}
    summary = {
        "weights": str(weights), "split": split, "imgsz": cfg["data"]["imgsz"],
        "precision": m.box.mp, "recall": m.box.mr,
        "mAP50": m.box.map50, "mAP50-95": m.box.map,
        "speed_ms_per_img": m.speed, "per_class": per_class,
    }
    (out_dir / f"{args.name}_{split}_metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/baseline.yaml"))
    ap.add_argument("--name", default="train", help="run name inside the output dir")
    ap.add_argument("--device", default=None, help="e.g. cpu, 0, 0,1 (default: auto)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--fraction", type=float, default=None, help="fraction of train set to use")
    ap.add_argument("--resume", action="store_true", help="resume the last run of --name")
    ap.add_argument("--eval-only", metavar="WEIGHTS", default=None,
                    help="skip training and evaluate these weights")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    out_dir = ROOT / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    split = SPLIT_ALIASES[cfg["eval"]["split"]]

    data_yaml = build_exdark_dataset()

    if args.eval_only:
        best = Path(args.eval_only)
    else:
        weights = cfg["model"]["weights"] or cfg["model"].get("yaml")
        if args.resume:
            last = out_dir / args.name / "weights/last.pt"
            YOLO(str(last)).train(resume=True)
        else:
            YOLO(weights).train(**train_args(cfg, data_yaml, out_dir, args))
        best = out_dir / args.name / "weights/best.pt"

    summary = evaluate(best, data_yaml, cfg, split, out_dir, args)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_class"}, indent=2))
    print(pd.DataFrame(summary["per_class"]).T.round(3).to_string())


if __name__ == "__main__":
    main()
