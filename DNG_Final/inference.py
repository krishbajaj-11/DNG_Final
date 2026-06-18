"""
=============================================================================
inference.py — Standalone Demo Script
Samsung SRIB-PRISM: Multi-Level Detail Enhancement Detection for RAW DNG Buffers
=============================================================================

QUICK START
-----------
1. Download model weights from Google Drive (link in README.md)
   Place best_model.pt inside:  checkpoints_v3/best_model.pt

2. Install dependencies:
   pip install -r requirements.txt

3. Run on a single image:
   python inference.py --image path/to/your/image.jpg

4. Run on a folder of images:
   python inference.py --folder path/to/folder/

5. Run on a DNG file:
   python inference.py --image path/to/photo.dng

OUTPUT EXAMPLE
--------------
  File     : sample.jpg
  Level    : L8  (Enhancement Level 8 / 10)
  Confidence: 94.3%
  Sharpness : 0.2841
  Latency  : 183.2 ms

  Level Probabilities:
    L5: 0.001   L6: 0.002   L7: 0.018
    L8: 0.943   L9: 0.031   L10: 0.005
=============================================================================
"""

import argparse
import sys
import time
import os
from pathlib import Path

import numpy as np
import torch
import cv2

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from model.classifier import load_model, build_model
from utils.metrics import compute_detail_score
from pipeline.dng_processor import DNGProcessor

# ── constants ────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = ROOT / "checkpoints_v3" / "best_model.pt"
IMG_SIZE           = 224
NUM_LEVELS         = 6
IMAGENET_MEAN      = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD       = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SUPPORTED_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".tiff",
                      ".dng", ".raw", ".nef", ".cr2", ".arw"}

LEVEL_DESCRIPTIONS = {
    5:  "Baseline — raw demosaiced, no enhancement",
    6:  "CLAHE contrast + mild Laplacian sharpening",
    7:  "Wavelet detail boost added",
    8:  "FFT-domain high-frequency enhancement",
    9:  "Retinex tonal correction",
    10: "Full stack — maximum detail enhancement",
}


# ── model loader ─────────────────────────────────────────────────────────────

def load(checkpoint_path: Path, device: str) -> torch.nn.Module:
    """Load checkpoint or fallback to untrained model for demo."""
    if checkpoint_path.exists():
        print(f"  ✅ Loading checkpoint: {checkpoint_path}")
        model = load_model(str(checkpoint_path), device=device)
    else:
        print(f"  ⚠️  Checkpoint not found at: {checkpoint_path}")
        print(f"     Falling back to UNTRAINED model (results will be random).")
        print(f"     Download weights from Google Drive — see README.md.\n")
        model = build_model(num_classes=NUM_LEVELS, pretrained=False)
    model.eval()
    return model.to(device)


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess(image_path: Path, device: str):
    """
    Load and preprocess an image (JPEG/PNG/DNG) into a model-ready tensor.
    Returns (tensor, raw_float32_image_for_metrics).
    """
    suffix = image_path.suffix.lower()

    if suffix in {".dng", ".raw", ".nef", ".cr2", ".arw"}:
        # RAW file → use LibRAW decoder
        processor = DNGProcessor()
        img = processor.process(str(image_path))          # float32 H×W×3, [0,1]
    else:
        # Standard image
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # 1×3×H×W
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.to(device), img


# ── single image inference ────────────────────────────────────────────────────

def classify_image(image_path: Path, model: torch.nn.Module, device: str) -> dict:
    """Run inference on a single image. Returns result dict."""
    t0 = time.perf_counter()

    tensor, img_np = preprocess(image_path, device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = int(probs.argmax().item())
        conf   = float(probs.max().item())

    level      = pred + 5                         # 0-indexed → L5..L10
    sharpness  = compute_detail_score(img_np)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "file":               image_path.name,
        "level":              level,
        "confidence":         round(conf * 100, 2),
        "sharpness":          round(float(sharpness), 4),
        "latency_ms":         round(elapsed_ms, 1),
        "level_probs":        {i + 5: round(float(p), 4) for i, p in enumerate(probs)},
        "description":        LEVEL_DESCRIPTIONS.get(level, ""),
    }


# ── pretty printer ────────────────────────────────────────────────────────────

def print_result(r: dict):
    bar = "█" * r["level"] + "░" * (10 - r["level"])
    print()
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │  File      : {r['file']:<30}│")
    print(f"  │  Level     : L{r['level']}  [{bar}]       │")
    print(f"  │  Description: {r['description']:<29}│")
    print(f"  │  Confidence: {r['confidence']:>5.1f}%                          │")
    print(f"  │  Sharpness : {r['sharpness']:<10.4f}                       │")
    print(f"  │  Latency   : {r['latency_ms']:>7.1f} ms                      │")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"  Level Probabilities:")
    for lv, p in r["level_probs"].items():
        bar_len = int(p * 30)
        print(f"    L{lv}: {'█' * bar_len}{'░' * (30 - bar_len)}  {p:.4f}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Samsung DNG Detail Level Classifier — Inference Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to a single image or DNG file")
    group.add_argument("--folder", type=str, help="Path to folder of images (batch mode)")

    parser.add_argument(
        "--checkpoint", type=str,
        default=str(DEFAULT_CHECKPOINT),
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})"
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on (default: auto-detect)"
    )
    parser.add_argument(
        "--save_csv", type=str, default=None,
        help="Optional: save batch results to a CSV file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  Samsung SRIB-PRISM: DNG Detail Level Classifier")
    print("  Model: ConvNeXt-Small | Accuracy: 98.53% | Kappa: 0.9968")
    print("=" * 60)
    print(f"  Device    : {args.device.upper()}")
    print(f"  Checkpoint: {args.checkpoint}")

    # Load model
    model = load(Path(args.checkpoint), args.device)

    results = []

    if args.image:
        # ── Single image mode ──
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"\n  ❌ File not found: {img_path}")
            sys.exit(1)
        print(f"\n  Running on: {img_path.name}")
        result = classify_image(img_path, model, args.device)
        print_result(result)
        results.append(result)

    else:
        # ── Batch folder mode ──
        folder = Path(args.folder)
        if not folder.is_dir():
            print(f"\n  ❌ Not a directory: {folder}")
            sys.exit(1)

        images = [f for f in sorted(folder.iterdir())
                  if f.suffix.lower() in SUPPORTED_EXTS]

        if not images:
            print(f"\n  ❌ No supported images found in: {folder}")
            sys.exit(1)

        print(f"\n  Found {len(images)} image(s) in: {folder}")
        print("  Processing...\n")

        for img_path in images:
            try:
                result = classify_image(img_path, model, args.device)
                print_result(result)
                results.append(result)
            except Exception as e:
                print(f"  ⚠️  Skipping {img_path.name}: {e}")

        # Summary
        if results:
            levels = [r["level"] for r in results]
            print(f"\n  ── Batch Summary ──────────────────────────────")
            print(f"  Images processed : {len(results)}")
            print(f"  Avg level        : L{sum(levels)/len(levels):.1f}")
            print(f"  Avg latency      : {sum(r['latency_ms'] for r in results)/len(results):.1f} ms")
            print(f"  ───────────────────────────────────────────────\n")

    # Optional CSV export
    if args.save_csv and results:
        import csv
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "level", "confidence",
                                                    "sharpness", "latency_ms", "description"])
            writer.writeheader()
            writer.writerows([{k: v for k, v in r.items() if k != "level_probs"}
                               for r in results])
        print(f"  ✅ Results saved to: {args.save_csv}\n")


if __name__ == "__main__":
    main()
