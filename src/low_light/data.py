"""ExDark -> YOLO-format dataset conversion shared by the baseline scripts."""
from pathlib import Path

import pandas as pd
from PIL import Image
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[2]
ANNO_DIR = ROOT / "data/raw/annotations/ExDark_Annno"
SPLIT_DIR = ROOT / "data/outputs/splits"

# Order defines the class ids of the 12-class ("exdark") scheme.
EXDARK_CLASSES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cup", "Dog", "Motorbike", "People", "Table",
]
# ExDark class -> COCO class id (for evaluating COCO-pretrained models as-is).
EXDARK_TO_COCO = {
    "Bicycle": 1, "Boat": 8, "Bottle": 39, "Bus": 5, "Car": 2, "Cat": 15,
    "Chair": 56, "Cup": 41, "Dog": 16, "Motorbike": 3, "People": 0, "Table": 60,
}
# Names of the splits as stored in data/outputs/splits/<name>.csv
SPLITS = ("train", "validation", "test")


def _annotation_file(cls: str, image_name: str) -> Path:
    """Annotation path for an image; tolerates extension-case mismatches (.JPG vs .jpg)."""
    exact = ANNO_DIR / cls / f"{image_name}.txt"
    if exact.exists():
        return exact
    stem = Path(image_name).stem
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        alt = ANNO_DIR / cls / f"{stem}{ext}.txt"
        if alt.exists():
            return alt
    raise FileNotFoundError(f"No annotation for {cls}/{image_name}")


def build_split(split: str, out_root: Path, class_ids: dict, force: bool = False) -> int:
    """Symlink the images of one split and write YOLO labels. Returns box count.

    class_ids maps ExDark class name -> integer id written to the label files.
    """
    img_out, lbl_out = out_root / split / "images", out_root / split / "labels"
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

        label_file = lbl_out / f"{Path(name).stem}.txt"
        if label_file.exists() and not force:
            n_boxes += len(label_file.read_text().splitlines())
            continue

        with Image.open(src) as im:
            w, h = im.size
        lines = []
        for row in _annotation_file(cls, src.name).read_text().splitlines():
            parts = row.split()
            if not parts or row.startswith("%") or parts[0] not in class_ids:
                continue
            x, y, bw, bh = map(float, parts[1:5])
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + bw, w), min(y + bh, h)
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(
                f"{class_ids[parts[0]]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            )
        n_boxes += len(lines)
        label_file.write_text("\n".join(lines))
    print(f"[{split}] {len(df)} images, {n_boxes} boxes")
    return n_boxes


def build_exdark_dataset(out_root: Path = ROOT / "data/yolo_exdark",
                         data_yaml: Path = ROOT / "data/exdark.yaml") -> Path:
    """12-class ExDark dataset (ids 0-11) for fine-tuning; returns the data yaml."""
    ids = {c: i for i, c in enumerate(EXDARK_CLASSES)}
    for split in SPLITS:
        build_split(split, out_root, ids)
    YAML.save(data_yaml, {
        "path": str(out_root),
        "train": "train/images", "val": "validation/images", "test": "test/images",
        "names": dict(enumerate(EXDARK_CLASSES)),
    })
    return data_yaml


def build_coco_split(split: str, coco_names: dict,
                     out_root: Path = ROOT / "data/yolo_coco") -> Path:
    """One split labelled with COCO ids, for zero-shot evaluation of COCO models."""
    build_split(split, out_root, EXDARK_TO_COCO)
    data_yaml = out_root / f"{split}.yaml"
    YAML.save(data_yaml, {
        "path": str(out_root), "train": f"{split}/images", "val": f"{split}/images",
        "names": coco_names,
    })
    return data_yaml
