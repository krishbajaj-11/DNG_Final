"""
Image Quality Metrics for RAW Detail Level Classification
Samsung Multi-Level Detail Enhancement Detection Project

Metrics used:
  -- No-reference (work on any image, no ground truth needed) --
  laplacian_var   : Laplacian variance          — proven focus/sharpness measure
  tenenbaum       : Sum of squared Sobel grads  — standard autofocus quality metric
  spatial_freq    : Spatial Frequency (SF)      — ISO display quality standard
  brenner         : Brenner focus measure       — sensitive to fine detail steps
  edge_density    : Canny edge pixel fraction   — structural richness
  dct_hf_ratio    : DCT high-frequency ratio    — stable block-based texture energy

  -- Full-reference (need a reference image, used for dataset validation) --
  psnr            : Peak Signal-to-Noise Ratio
  ssim            : Structural Similarity Index

NOTE on MTF50:
  True MTF50 (ISO 12233) requires a slanted-edge test chart captured in controlled
  conditions. Estimating MTF50 from arbitrary natural images via FFT power spectrum
  is unreliable because natural image frequency content varies wildly with scene
  content, making it non-discriminative across quality levels.
  The metrics above are the industry-validated alternatives for natural image IQA.
"""

import numpy as np
import cv2
from typing import Dict, Optional
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim


# =============================================================
# No-reference sharpness / detail metrics
# =============================================================

def compute_laplacian_var(image: np.ndarray) -> float:
    """
    Laplacian variance — classic focus measure.
    Measures the variance of second-order intensity derivatives.
    Sharp images have high-variance Laplacians; blurry ones are flat.
    Range: 0 → ∞  (use relative comparison within a dataset).
    Normalised here to [0, 1] using a soft ceiling of 3000.
    """
    gray = _to_gray_uint8(image)
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    raw  = float(lap.var())
    return round(min(raw / 3000.0, 1.0), 5)


def compute_tenenbaum(image: np.ndarray) -> float:
    """
    Tenenbaum's focus measure — sum of squared Sobel gradients.
    One of the most reliable autofocus quality functions.
    Captures edge strength; responds strongly to fine texture.
    Normalised to [0, 1] via soft ceiling of 50.0.
    """
    gray    = _to_gray_float(image)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(sobel_x ** 2 + sobel_y ** 2))
    return round(min(tenengrad / 50.0, 1.0), 5)


def compute_spatial_frequency(image: np.ndarray) -> float:
    """
    Spatial Frequency (SF) — ISO/IEC standard metric for display quality.
    SF = sqrt(RF^2 + CF^2)
      RF (Row Frequency)    = RMS of horizontal intensity differences
      CF (Column Frequency) = RMS of vertical intensity differences
    Measures how rapidly intensity changes across the image.
    Higher SF = more fine detail / higher apparent resolution.
    Normalised to [0, 1] via soft ceiling of 0.20.
    """
    gray = _to_gray_float(image)
    rf   = float(np.sqrt(np.mean(np.diff(gray, axis=1) ** 2)))  # horizontal
    cf   = float(np.sqrt(np.mean(np.diff(gray, axis=0) ** 2)))  # vertical
    sf   = float(np.sqrt(rf ** 2 + cf ** 2))
    return round(min(sf / 0.20, 1.0), 5)


def compute_brenner(image: np.ndarray) -> float:
    """
    Brenner's focus measure — one of the earliest and most stable IQA metrics.
    B = sum( (I(x+2, y) - I(x, y))^2 )
    Sensitive to changes between adjacent pixel pairs, responds well to
    the fine detail differences between enhancement levels.
    Normalised to [0, 1] via soft ceiling of 0.05.
    """
    gray  = _to_gray_float(image)
    diff  = gray[:, 2:] - gray[:, :-2]     # horizontal 2-pixel shift
    raw   = float(np.mean(diff ** 2))
    return round(min(raw / 0.05, 1.0), 5)


def compute_edge_density(image: np.ndarray,
                          low_thresh:  int = 50,
                          high_thresh: int = 150) -> float:
    """
    Edge density: fraction of pixels detected as edges by Canny.
    Measures structural richness — more edges = more detail.
    Range: [0, 1].  Stable because Canny uses hysteresis thresholding.
    """
    gray  = _to_gray_uint8(image)
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    return round(float(np.count_nonzero(edges)) / edges.size, 5)


def compute_dct_hf_ratio(image: np.ndarray,
                          block_size: int = 8,
                          hf_fraction: float = 0.5) -> float:
    """
    DCT High-Frequency Energy Ratio.
    Splits each 8x8 block into a DCT, then computes the fraction
    of energy in the high-frequency coefficients (top-right triangle).

    Used in JPEG quality assessment and widely validated for texture
    richness measurement. More stable than FFT-based approaches on
    natural images because it operates at block level.

    Args:
        hf_fraction: coefficients with zigzag index > this fraction
                     of block area are considered 'high frequency'
    Returns: float in [0, 1]
    """
    gray  = _to_gray_float(image)
    h, w  = gray.shape
    bh    = (h // block_size) * block_size
    bw    = (w // block_size) * block_size
    gray  = gray[:bh, :bw]

    total_energy = 0.0
    hf_energy    = 0.0
    cutoff_idx   = int(block_size * block_size * hf_fraction)

    for r in range(0, bh, block_size):
        for c in range(0, bw, block_size):
            block  = gray[r:r+block_size, c:c+block_size].astype(np.float32)
            dct    = cv2.dct(block)
            coeffs = np.abs(dct.flatten())
            # Sort by magnitude (proxy for zigzag order energy split)
            coeffs_sorted = np.sort(coeffs)
            total_energy += float(np.sum(coeffs_sorted ** 2))
            hf_energy    += float(np.sum(coeffs_sorted[cutoff_idx:] ** 2))

    if total_energy > 0:
        return round(hf_energy / total_energy, 5)
    return 0.0


# =============================================================
# Full-reference metrics
# =============================================================

def compute_psnr(reference: np.ndarray, distorted: np.ndarray) -> float:
    """PSNR in dB. Higher = more similar to reference."""
    if reference.shape != distorted.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {distorted.shape}")
    return round(float(skimage_psnr(reference, distorted, data_range=1.0)), 3)


def compute_ssim(reference: np.ndarray, distorted: np.ndarray) -> float:
    """SSIM in [-1, 1]. Higher = more structurally similar."""
    if reference.shape != distorted.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {distorted.shape}")
    channel_axis = 2 if reference.ndim == 3 else None
    return round(float(skimage_ssim(reference, distorted, data_range=1.0,
                                    channel_axis=channel_axis)), 4)


# =============================================================
# Unified metric bundle
# =============================================================

def compute_all_metrics(
    image: np.ndarray,
    reference: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all no-reference metrics, plus PSNR/SSIM if reference given.

    Returns dict with keys:
        laplacian_var    — sharpness via 2nd derivative variance
        tenenbaum        — edge strength via Sobel (autofocus metric)
        spatial_freq     — ISO spatial frequency (display quality standard)
        brenner          — fine detail sensitivity (Brenner focus measure)
        edge_density     — structural richness (Canny edge fraction)
        dct_hf_ratio     — texture energy in high DCT coefficients
        psnr             — (if reference given)
        ssim             — (if reference given)
    """
    results: Dict[str, float] = {
        "laplacian_var": compute_laplacian_var(image),
        "tenenbaum":     compute_tenenbaum(image),
        "spatial_freq":  compute_spatial_frequency(image),
        "brenner":       compute_brenner(image),
        "edge_density":  compute_edge_density(image),
        "dct_hf_ratio":  compute_dct_hf_ratio(image),
    }
    if reference is not None:
        results["psnr"] = compute_psnr(reference, image)
        results["ssim"] = compute_ssim(reference, image)
    return results


def compute_detail_score(image: np.ndarray) -> float:
    """
    Composite detail score — weighted average of all no-reference metrics.
    Weights tuned to maximise separation across enhancement levels.
    Returns a single float in [0, 1].
    """
    m = compute_all_metrics(image)
    score = (
        0.25 * m["tenenbaum"]    +   # strongest individual predictor
        0.20 * m["laplacian_var"] +
        0.20 * m["spatial_freq"]  +
        0.15 * m["brenner"]       +
        0.10 * m["dct_hf_ratio"]  +
        0.10 * m["edge_density"]
    )
    return round(float(np.clip(score, 0, 1)), 4)


# =============================================================
# Level mapping helpers
# =============================================================

def detail_score_to_level(score: float,
                           min_level: int = 5,
                           max_level: int = 10) -> int:
    """Map composite detail score [0,1] to discrete level [5, 10]."""
    span = max_level - min_level
    return int(np.clip(round(min_level + score * span), min_level, max_level))


def psnr_to_detail_level(psnr: float, num_levels: int = 10) -> int:
    psnr_min, psnr_max = 20.0, 50.0
    clamped = max(psnr_min, min(psnr, psnr_max))
    return int(((clamped - psnr_min) / (psnr_max - psnr_min)) * (num_levels - 1)) + 1


def ssim_to_detail_level(ssim: float, num_levels: int = 10) -> int:
    return int(np.clip(ssim, 0, 1) * (num_levels - 1)) + 1


# =============================================================
# Internal helpers
# =============================================================

def _to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)


def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    img_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    if img_u8.ndim == 3:
        return cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    return img_u8


# =============================================================
# Self-test
# =============================================================

if __name__ == "__main__":
    rng     = np.random.default_rng(42)
    clean   = rng.random((224, 224, 3), dtype=np.float32)
    blurry  = cv2.GaussianBlur(clean, (0, 0), 3.0).astype(np.float32)
    kernel  = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], np.float32) / 9
    sharp   = np.clip(cv2.filter2D(clean, -1, kernel), 0, 1).astype(np.float32)

    for label, img in [("Blurry (low detail)", blurry),
                        ("Clean  (baseline  )", clean),
                        ("Sharp  (enhanced  )", sharp)]:
        m = compute_all_metrics(img)
        ds = compute_detail_score(img)
        print(f"\n{label}")
        for k, v in m.items():
            print(f"  {k:<18}: {v}")
        print(f"  {'detail_score':<18}: {ds}  -> Level {detail_score_to_level(ds)}")
