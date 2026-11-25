#!/usr/bin/env python3
"""One-shot, cross-platform project setup (Linux / macOS / Windows).

    python setup.py

Uses only the standard library so it can run before any dependency is installed.
Idempotent: every step is skipped if its output already exists.

Steps: uv -> dependencies -> ExDark images + annotations -> splits ->
YOLO-format dataset -> yolov8n.pt -> sanity check.

Afterwards:
    uv run python scripts/baseline_eval.py          # zero-shot YOLOv8n, test split
    uv run python scripts/train.py --device 0       # fine-tune + evaluate
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def step(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)   # scripts use paths relative to the root


def find_uv() -> list[str]:
    """Return the command that invokes uv, installing it with pip if necessary."""
    exe = shutil.which("uv")
    if exe:
        return [exe]
    step("uv not found - installing it with pip (--user)")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "uv"], check=True)
        subprocess.run([sys.executable, "-m", "uv", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError):
        sys.exit("Could not install uv automatically. Install it manually: "
                 "https://docs.astral.sh/uv/getting-started/installation/")
    return [sys.executable, "-m", "uv"]


def main() -> None:
    uv = find_uv()
    uv_py = uv + ["run", "python"]

    step("Installing dependencies (uv sync)")
    try:
        run(uv + ["sync", "--frozen"])          # exact versions from uv.lock
    except subprocess.CalledProcessError:
        run(uv + ["sync"])                      # lockfile out of date: re-resolve

    step("Downloading ExDark images")
    run(uv_py + ["scripts/download_exdark.py"])

    step("Downloading ExDark annotations")
    run(uv_py + ["scripts/download_exdark_groundtruth.py"])

    step("Preparing class list and train/validation/test splits")
    run(uv_py + ["scripts/prepare_dataset.py"])

    step("Building YOLO-format dataset")
    run(uv_py + ["-c", "from low_light.data import build_exdark_dataset; build_exdark_dataset()"])

    if (ROOT / "yolov8n.pt").exists():
        step("yolov8n.pt already present")
    else:
        step("Downloading yolov8n.pt")
        run(uv_py + ["-c", "from ultralytics import YOLO; YOLO('yolov8n.pt')"])

    step("Verifying")
    run(uv_py + ["-c", (
        "from pathlib import Path\n"
        "import torch\n"
        "import ultralytics, low_light.data\n"
        "need = ['data/outputs/splits/train.csv', 'data/outputs/splits/validation.csv',\n"
        "        'data/outputs/splits/test.csv', 'data/exdark.yaml', 'yolov8n.pt']\n"
        "missing = [p for p in need if not Path(p).exists()]\n"
        "assert not missing, f'missing: {missing}'\n"
        "print('ultralytics', ultralytics.__version__, '| CUDA available:', torch.cuda.is_available())\n"
    )])

    print("\nSetup complete. Run:\n"
          "  uv run python scripts/baseline_eval.py       # zero-shot baseline (test split)\n"
          "  uv run python scripts/train.py --device 0    # fine-tune + evaluate (--device cpu without a GPU)")


if __name__ == "__main__":
    main()
