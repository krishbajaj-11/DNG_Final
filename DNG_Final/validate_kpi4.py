"""PSNR/SSIM Correlation KPI validation — KPI target: |r| >= 0.9"""
import sys, json, random
sys.path.insert(0, '.')
import numpy as np
import cv2
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

DATA_DIR = Path('data/fivek_prepared')

with open(DATA_DIR / 'labels.json') as f:
    labels = json.load(f)

names = list(set(p.split('/')[-1] for p in labels.keys()))
random.seed(42)
sample = random.sample(names, min(200, len(names)))

pred_levels, actual_psnrs, actual_ssims = [], [], []

for name in sample:
    l5_path = DATA_DIR / 'level_05' / name
    if not l5_path.exists():
        continue
    ref = cv2.imread(str(l5_path))
    if ref is None:
        continue
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    for lv in range(5, 11):
        path = DATA_DIR / f'level_{lv:02d}' / name
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        p = psnr(ref, img, data_range=1.0) if lv > 5 else 60.0
        s = ssim(ref, img, data_range=1.0, channel_axis=2)

        pred_levels.append(lv)
        actual_psnrs.append(p)
        actual_ssims.append(s)

pred  = np.array(pred_levels)
psnrs = np.array(actual_psnrs)
ssims = np.array(actual_ssims)

r_psnr,  _ = spearmanr(pred, psnrs)
r_ssim,  _ = spearmanr(pred, ssims)
rp_psnr, _ = pearsonr(pred, psnrs)
rp_ssim, _ = pearsonr(pred, ssims)

print('=' * 55)
print('PSNR/SSIM CORRELATION KPI  (target: |r| >= 0.9)')
print('=' * 55)
for name, val in [
    ('Spearman  Level vs PSNR', r_psnr),
    ('Spearman  Level vs SSIM', r_ssim),
    ('Pearson   Level vs PSNR', rp_psnr),
    ('Pearson   Level vs SSIM', rp_ssim),
]:
    status = 'PASS' if abs(val) >= 0.9 else 'FAIL'
    print(f'{name}: {val:+.4f}  |r|={abs(val):.4f}  [{status}]')

print(f'\nSamples tested: {len(pred_levels)}')
