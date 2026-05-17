"""
DNG RAW Buffer Extraction and Preprocessing Pipeline
Samsung Multi-Level Detail Enhancement Detection Project
"""

import rawpy
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DNGProcessor:
    """
    Handles reading DNG files, extracting RAW buffers,
    and preprocessing them into model-ready tensors.
    """

    TARGET_SIZE = (512, 512)

    def __init__(self, use_demosaic: bool = True):
        self.use_demosaic = use_demosaic

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, dng_path: str) -> np.ndarray:
        """
        Full pipeline: DNG file → preprocessed float32 RGB image (H, W, 3).
        Values are normalized to [0, 1].
        """
        raw_buffer = self.extract_raw_buffer(dng_path)
        processed = self.preprocess(raw_buffer, dng_path)
        return processed

    def extract_raw_buffer(self, dng_path: str) -> np.ndarray:
        """
        Extract the raw Bayer array from a DNG file.
        Returns float32 array normalized to [0, 1].
        """
        path = Path(dng_path)
        if not path.exists():
            raise FileNotFoundError(f"DNG file not found: {dng_path}")
        if path.suffix.lower() not in (".dng", ".raw", ".nef", ".cr2", ".arw"):
            raise ValueError(f"Unsupported file format: {path.suffix}")

        with rawpy.imread(str(path)) as raw:
            bayer = raw.raw_image_visible.copy().astype(np.float32)
            black = raw.black_level_per_channel
            white = raw.white_level

            # Black-level subtraction and white-level normalization
            black_mean = float(np.mean(black)) if black is not None else 0.0
            bayer = np.clip(bayer - black_mean, 0, None)
            if white and white > 0:
                bayer /= (white - black_mean)
            else:
                bayer /= bayer.max() if bayer.max() > 0 else 1.0

        logger.debug("Extracted RAW buffer shape: %s", bayer.shape)
        return bayer

    def preprocess(
        self,
        raw_buffer: np.ndarray,
        dng_path: Optional[str] = None,
        target_size: Tuple[int, int] = None,
    ) -> np.ndarray:
        """
        Convert a raw Bayer buffer to a preprocessed RGB image
        suitable for the classifier.
        """
        size = target_size or self.TARGET_SIZE

        if self.use_demosaic and dng_path is not None:
            rgb = self._demosaic_with_rawpy(dng_path)
        else:
            rgb = self._bayer_to_rgb_simple(raw_buffer)

        # Resize
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)

        # CLAHE on luminance channel for local contrast enhancement
        rgb = self._apply_clahe(rgb)

        # Final normalization to [0, 1]
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        return rgb  # shape: (H, W, 3)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _demosaic_with_rawpy(self, dng_path: str) -> np.ndarray:
        """Use LibRAW's built-in demosaicing for highest quality."""
        with rawpy.imread(dng_path) as raw:
            rgb16 = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=True,
                output_bps=16,
            )
        rgb = rgb16.astype(np.float32) / 65535.0
        return rgb

    def _bayer_to_rgb_simple(self, bayer: np.ndarray) -> np.ndarray:
        """
        Simple RGGB Bayer → RGB conversion using OpenCV.
        Fallback when no DNG path is available.
        """
        bayer_uint16 = (bayer * 65535).astype(np.uint16)
        rgb16 = cv2.cvtColor(bayer_uint16, cv2.COLOR_BayerRG2RGB)
        return rgb16.astype(np.float32) / 65535.0

    def _apply_clahe(self, rgb: np.ndarray) -> np.ndarray:
        """Apply CLAHE on the L channel of LAB color space."""
        rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
        lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        rgb_out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return rgb_out.astype(np.float32) / 255.0


# ------------------------------------------------------------------
# Quick smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) < 2:
        print("Usage: python dng_processor.py <path_to_dng>")
        sys.exit(1)

    proc = DNGProcessor()
    img = proc.process(sys.argv[1])
    print(f"Output shape: {img.shape}, dtype: {img.dtype}")
    print(f"Min: {img.min():.4f}, Max: {img.max():.4f}")

    # Save preview
    preview = (img * 255).astype(np.uint8)
    cv2.imwrite("preview.png", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    print("Saved preview.png")
