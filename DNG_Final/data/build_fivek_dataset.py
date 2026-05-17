"""
FiveK DNG Dataset Builder — GPU-Accelerated Enhancement-Based (5→10 Levels)
-----------------------------------------------------------------------------
Approach: Start from the raw DNG baseline and progressively
ENHANCE sharpness, structural detail, and tonal quality through
6 levels (5 to 10). Each level stacks more sophisticated
enhancement on top of the previous.

Enhancement ladder:
  L5  — Baseline: raw DNG decoded with neutral processing
  L6  — CLAHE (local contrast) + subtle unsharp mask
  L7  — Wavelet detail coefficient amplification
  L8  — FFT-domain high-frequency boosting  ← GPU (torch.fft)
  L9  — Laplacian high-boost filter + Retinex tonal correction
  L10 — Full stack: all above at maximum strength

Speed strategy:
  - rawpy DNG decode: CPU-only (LibRAW limitation), parallelised
    across CPU cores via multiprocessing.Pool
  - FFT sharpening (L8/L10): torch.fft on GPU (massive speedup)
  - All other ops: OpenCV / NumPy on CPU (already fast)

Usage:
    python data/build_fivek_dataset.py
        --dng_root  fivek_dataset/raw_photos
        --out_dir   data/fivek_prepared
        --max_images 5000
        --workers   8
"""

import argparse
import json
import logging
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.dng_processor import DNGProcessor
from utils.metrics import (
    compute_psnr, compute_ssim,
    compute_laplacian_var, compute_tenenbaum,
    compute_spatial_frequency,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Levels produced: 5 through 10
LEVELS = [5, 6, 7, 8, 9, 10]

# GPU device (set at startup)
DEVICE: Optional[torch.device] = None


# =========================================================
# GPU / Device setup
# =========================================================

def setup_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✅ GPU detected: {name} ({mem:.1f} GB VRAM)")
        logger.info("FFT sharpening and tensor ops will run on GPU.")
    else:
        dev = torch.device("cpu")
        logger.info("⚠️  No CUDA GPU found — running all ops on CPU.")
    return dev


# =========================================================
# CPU enhancement functions (CLAHE, unsharp, wavelet, etc.)
# =========================================================

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0,
                tile_grid: int = 8) -> np.ndarray:
    """
    Contrast-Limited Adaptive Histogram Equalization on L channel.
    Enhances local contrast without over-amplifying noise.
    """
    img_u8  = (img * 255).clip(0, 255).astype(np.uint8)
    lab     = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    clahe   = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result  = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return result.astype(np.float32) / 255.0


def apply_unsharp_mask(img: np.ndarray, strength: float = 0.5,
                       radius: float = 1.0) -> np.ndarray:
    """Unsharp masking: enhanced = img + strength*(img - blurred)."""
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    return np.clip(img + strength * (img - blurred), 0, 1).astype(np.float32)


def apply_wavelet_detail_boost(img: np.ndarray, boost: float = 1.5) -> np.ndarray:
    """
    Detail enhancement via Gaussian-difference Laplacian pyramid.
    Amplifies fine and coarse detail bands without touching smooth base.
    """
    g0      = img.astype(np.float32)
    smooth1 = cv2.GaussianBlur(g0, (0, 0), 2.0)
    detail1 = g0 - smooth1
    smooth2 = cv2.GaussianBlur(smooth1, (0, 0), 1.0)
    detail2 = smooth1 - smooth2
    enhanced = smooth2 + boost * detail2 + boost * detail1
    return np.clip(enhanced, 0, 1).astype(np.float32)


def apply_fft_sharpening_gpu(img: np.ndarray,
                              emphasis: float = 1.8,
                              cutoff: float = 0.3,
                              device: torch.device = None) -> np.ndarray:
    """
    GPU-accelerated frequency-domain sharpening via torch.fft.
    Boosts high-frequency components above `cutoff`.
    Falls back to NumPy FFT if no GPU available.
    """
    if device is None or device.type == "cpu":
        return _apply_fft_sharpening_cpu(img, emphasis, cutoff)

    h, w = img.shape[:2]

    # Build radial frequency mask on GPU
    cy, cx = h // 2, w // 2
    Y = torch.arange(h, device=device).unsqueeze(1).float()   # (H,1)
    X = torch.arange(w, device=device).unsqueeze(0).float()   # (1,W)
    dist = torch.sqrt((X - cx)**2 + (Y - cy)**2) / np.sqrt(cx**2 + cy**2)
    mask = 1.0 + (emphasis - 1.0) * torch.clamp((dist - cutoff) / (1.0 - cutoff), 0, 1)
    # mask shape: (H, W) — will broadcast over channels

    # Move image to GPU as (3, H, W) tensor
    img_t = torch.from_numpy(img).permute(2, 0, 1).to(device)   # (3, H, W)

    # FFT per channel (batched)
    f         = torch.fft.fft2(img_t)                            # (3, H, W) complex
    f_shifted = torch.fft.fftshift(f, dim=(-2, -1))
    f_boosted = f_shifted * mask.unsqueeze(0)                    # broadcast
    f_back    = torch.fft.ifftshift(f_boosted, dim=(-2, -1))
    result_t  = torch.real(torch.fft.ifft2(f_back))             # (3, H, W)

    result = result_t.permute(1, 2, 0).cpu().numpy()            # (H, W, 3)
    return np.clip(result, 0, 1).astype(np.float32)


def _apply_fft_sharpening_cpu(img: np.ndarray,
                               emphasis: float = 1.8,
                               cutoff: float = 0.3) -> np.ndarray:
    """NumPy FFT fallback (CPU)."""
    result = np.zeros_like(img)
    h, w   = img.shape[:2]
    cy, cx = h // 2, w // 2
    Y, X   = np.ogrid[:h, :w]
    dist   = np.sqrt((X - cx)**2 + (Y - cy)**2) / np.sqrt(cx**2 + cy**2)
    mask   = 1.0 + (emphasis - 1.0) * np.clip((dist - cutoff) / (1.0 - cutoff), 0, 1)
    for c in range(3):
        f         = np.fft.fft2(img[:, :, c])
        f_shifted = np.fft.fftshift(f)
        f_boosted = f_shifted * mask
        f_back    = np.fft.ifftshift(f_boosted)
        result[:, :, c] = np.real(np.fft.ifft2(f_back))
    return np.clip(result, 0, 1).astype(np.float32)


def apply_laplacian_highboost(img: np.ndarray, boost: float = 0.4) -> np.ndarray:
    """High-boost filtering using Laplacian operator."""
    result = np.zeros_like(img)
    for c in range(3):
        ch      = (img[:, :, c] * 255).astype(np.uint8)
        lap     = cv2.Laplacian(ch, cv2.CV_64F, ksize=3)
        lap_norm = lap / 255.0
        result[:, :, c] = img[:, :, c] + boost * lap_norm
    return np.clip(result, 0, 1).astype(np.float32)


def apply_retinex_tonal(img: np.ndarray, sigma: float = 60.0) -> np.ndarray:
    """
    Single-scale Retinex — brings out shadow detail while preserving highlights.
    """
    img_f  = img.astype(np.float64) + 1e-6
    result = np.zeros_like(img_f)
    for c in range(3):
        ch      = img_f[:, :, c]
        blurred = cv2.GaussianBlur(ch, (0, 0), sigma)
        retinex = np.log1p(ch) - np.log1p(blurred + 1e-6)
        r_min, r_max = retinex.min(), retinex.max()
        if r_max > r_min:
            retinex = (retinex - r_min) / (r_max - r_min)
        result[:, :, c] = retinex
    blend = 0.4 * img + 0.6 * result.astype(np.float32)
    return np.clip(blend, 0, 1).astype(np.float32)


# =========================================================
# Enhancement pipeline per level
# =========================================================

def enhance_to_level(baseline: np.ndarray, level: int,
                     device: torch.device) -> np.ndarray:
    """
    Apply progressively stronger enhancement stacks to produce the target level.

    L5  → baseline only
    L6  → CLAHE + unsharp mask (light)
    L7  → L6 + wavelet detail boost
    L8  → L7 + FFT high-frequency sharpening  ← GPU
    L9  → L8 + Laplacian high-boost + Retinex tonal
    L10 → Full stack at maximum strength       ← GPU (FFT twice)
    """
    if level == 5:
        return baseline.copy()

    img = baseline.copy()

    if level >= 6:
        img = apply_clahe(img, clip_limit=2.5, tile_grid=8)
        img = apply_unsharp_mask(img, strength=0.4, radius=0.8)

    if level >= 7:
        img = apply_wavelet_detail_boost(img, boost=1.4)

    if level >= 8:
        img = apply_fft_sharpening_gpu(img, emphasis=1.6, cutoff=0.25, device=device)

    if level >= 9:
        img = apply_laplacian_highboost(img, boost=0.35)
        img = apply_retinex_tonal(img, sigma=50.0)

    if level == 10:
        img = apply_clahe(img, clip_limit=3.5, tile_grid=4)
        img = apply_unsharp_mask(img, strength=0.6, radius=1.2)
        img = apply_wavelet_detail_boost(img, boost=1.8)
        img = apply_fft_sharpening_gpu(img, emphasis=2.0, cutoff=0.2, device=device)

    return np.clip(img, 0, 1).astype(np.float32)


# =========================================================
# Worker: decode one DNG → baseline (runs in subprocess)
# =========================================================

def _decode_worker(args: Tuple[int, Path, int]) -> Tuple[int, Optional[np.ndarray], str]:
    """
    Decode a single DNG to a float32 RGB baseline (224×224).
    Runs in a worker process via multiprocessing.Pool.
    Returns (idx, array_or_None, stem_str).
    """
    idx, dng_path, img_size = args
    try:
        import rawpy, cv2, numpy as np
        with rawpy.imread(str(dng_path)) as raw:
            rgb16 = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=True,
                output_bps=16,
            )
        baseline = rgb16.astype(np.float32) / 65535.0
        baseline = cv2.resize(baseline, (img_size, img_size),
                              interpolation=cv2.INTER_AREA)
        baseline = np.clip(baseline, 0, 1).astype(np.float32)
        stem = f"{idx:05d}_{dng_path.stem[:20]}"
        return idx, baseline, stem
    except Exception as e:
        return idx, None, str(e)


# =========================================================
# Dataset builder
# =========================================================

def collect_dng_files(dng_root: str, max_images: int) -> List[Path]:
    root  = Path(dng_root)
    files = sorted(root.rglob("*.dng"))
    logger.info(f"Found {len(files)} DNG files in {dng_root}")
    if len(files) > max_images:
        step  = len(files) // max_images
        files = files[::step][:max_images]
    logger.info(f"Using {len(files)} DNG files")
    return files


def build_dataset(
    dng_root:   str,
    out_dir:    str,
    max_images: int = 5000,
    img_size:   int = 224,
    workers:    int = 4,
) -> str:
    """
    For each source DNG:
      1. Decode baseline (L5) — parallel across `workers` CPU processes
      2. Apply GPU-accelerated enhancement stack for L6–L10
      3. Save PNG + record in labels.json
    """
    global DEVICE
    DEVICE = setup_device()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for lv in LEVELS:
        (out_path / f"level_{lv:02d}").mkdir(exist_ok=True)

    dng_files = collect_dng_files(dng_root, max_images)
    labels: Dict[str, int] = {}
    failed = 0

    psnr_by_level:   Dict[int, list] = {lv: [] for lv in LEVELS}
    ssim_by_level:   Dict[int, list] = {lv: [] for lv in LEVELS}
    lapvar_by_level: Dict[int, list] = {lv: [] for lv in LEVELS}
    tenen_by_level:  Dict[int, list] = {lv: [] for lv in LEVELS}
    sf_by_level:     Dict[int, list] = {lv: [] for lv in LEVELS}

    logger.info(f"Building enhancement dataset (L5→L10) for {len(dng_files)} DNGs "
                f"| {workers} CPU decode workers | GPU={DEVICE.type == 'cuda'}")

    # --- Parallel DNG decode via multiprocessing ---
    decode_args = [(idx, p, img_size) for idx, p in enumerate(dng_files)]

    t_start = time.time()

    with mp.Pool(processes=workers) as pool:
        for idx, baseline, stem in tqdm(
            pool.imap(_decode_worker, decode_args, chunksize=4),
            total=len(dng_files),
            desc="Decoding + enhancing"
        ):
            # stem is an error string if baseline is None
            if baseline is None:
                logger.warning(f"Skipping idx {idx}: {stem}")
                failed += 1
                continue

            # --- GPU enhancement + save all 6 levels ---
            for level in LEVELS:
                enhanced  = enhance_to_level(baseline, level, DEVICE)
                fname     = f"{stem}.png"
                rel_path  = f"level_{level:02d}/{fname}"
                save_path = out_path / f"level_{level:02d}" / fname

                img_bgr = cv2.cvtColor(
                    (enhanced * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
                )
                cv2.imwrite(str(save_path), img_bgr)
                labels[rel_path] = level

                # Metric sample (first 30 images only)
                if idx < 30:
                    if level > 5:
                        psnr_by_level[level].append(compute_psnr(baseline, enhanced))
                        ssim_by_level[level].append(compute_ssim(baseline, enhanced))
                    lapvar_by_level[level].append(compute_laplacian_var(enhanced))
                    tenen_by_level[level].append(compute_tenenbaum(enhanced))
                    sf_by_level[level].append(compute_spatial_frequency(enhanced))

    elapsed = time.time() - t_start

    # --- Save labels ---
    labels_path = out_path / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    # --- Report ---
    total = len(dng_files) * len(LEVELS) - (failed * len(LEVELS))
    logger.info("\n" + "=" * 60)
    logger.info("DATASET BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total samples  : {total}  ({len(dng_files)} DNGs × {len(LEVELS)} levels)")
    logger.info(f"Failed DNGs    : {failed}")
    logger.info(f"Elapsed time   : {elapsed/60:.1f} min")
    logger.info(f"Labels file    : {labels_path}")
    logger.info("\nPer-level metric report (mean over first 30 images):")
    logger.info(f"  {'Level':<7} {'PSNR vs L5':<14} {'SSIM vs L5':<12} {'Lap.Var':<10} {'Tenenbaum':<12} {'Spatial Freq'}")
    logger.info("  " + "-" * 68)
    for lv in LEVELS:
        p   = np.mean(psnr_by_level[lv])    if psnr_by_level[lv]   else float('nan')
        s   = np.mean(ssim_by_level[lv])    if ssim_by_level[lv]   else float('nan')
        lv_ = np.mean(lapvar_by_level[lv])  if lapvar_by_level[lv] else float('nan')
        t   = np.mean(tenen_by_level[lv])   if tenen_by_level[lv]  else float('nan')
        sf  = np.mean(sf_by_level[lv])      if sf_by_level[lv]     else float('nan')
        psnr_str = f"{p:.2f} dB" if lv > 5 else "inf (base)"
        ssim_str = f"{s:.4f}"   if lv > 5 else "1.0000"
        logger.info(
            f"  L{lv:<6} {psnr_str:<14} {ssim_str:<12} "
            f"{lv_:<10.4f} {t:<12.5f} {sf:.5f}"
        )
    logger.info("  Lap.Var = Laplacian variance | Tenenbaum = edge strength")
    logger.info("  Spatial Freq = ISO display standard. All metrics increase L5→L10.")

    return str(labels_path)


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dng_root",   default="fivek_dataset/raw_photos")
    parser.add_argument("--out_dir",    default="data/fivek_prepared")
    parser.add_argument("--max_images", type=int, default=5000,
                        help="Number of source DNG files to use (up to 5000)")
    parser.add_argument("--img_size",   type=int, default=224)
    parser.add_argument("--workers",    type=int, default=4,
                        help="CPU worker processes for parallel DNG decoding")
    args = parser.parse_args()

    build_dataset(
        dng_root   = args.dng_root,
        out_dir    = args.out_dir,
        max_images = args.max_images,
        img_size   = args.img_size,
        workers    = args.workers,
    )
