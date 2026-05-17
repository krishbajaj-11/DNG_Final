"""\nTraining Script — v3 (ConvNeXt-Small + 384px)
Samsung Multi-Level Detail Enhancement Detection Project

Key upgrades over v1:
  1. Label smoothing (0.1) — stops punishing near-miss level confusion
  2. Progressive backbone unfreeze — epoch 1-5 head only, then full fine-tune
  3. Stronger augmentation (via dataset.py flags)
  4. CosineAnnealingWarmRestarts — escapes flat plateaus with LR resets
  5. Longer training (50 epochs) with early stopping

Usage:
    python model/train.py --data_dir data/fivek_prepared --num_levels 6
"""

import argparse
import time
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import cohen_kappa_score, accuracy_score
import numpy as np

from model.classifier import build_model, OrdinalCrossEntropyLoss, save_model
from data.dataset import create_dataloaders, generate_synthetic_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Backbone freeze / unfreeze helpers
# ------------------------------------------------------------------

def freeze_backbone(model):
    """Freeze EfficientNet feature extractor — only train the head."""
    for p in model.features.parameters():
        p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Backbone FROZEN — {trainable:,} trainable params (head only)")


def unfreeze_backbone(model):
    """Unfreeze everything for full fine-tuning."""
    for p in model.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Backbone UNFROZEN — {trainable:,} trainable params (full model)")


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss   = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds      = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss   = loss_fn(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n     = len(all_labels)
    acc   = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return total_loss / n, acc, kappa


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info("=== v2 Training: Label smoothing + Progressive unfreeze + Warm restarts ===")

    # --- Data ---
    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not (data_dir / "labels.json").exists():
        logger.info("No dataset found. Generating synthetic dataset...")
        generate_synthetic_dataset(
            str(data_dir),
            num_images=args.synthetic_images,
            num_levels=args.num_levels,
        )

    train_loader, val_loader, test_loader = create_dataloaders(
        str(data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        num_levels=args.num_levels,
    )

    # --- Model ---
    model = build_model(num_classes=args.num_levels, pretrained=True).to(device)

    # FIX 1: Label smoothing (0.1) — key fix for adjacent-level confusion
    # Ordinal loss still applied, but CE component uses smoothed labels
    loss_fn = OrdinalCrossEntropyLoss(
        num_classes=args.num_levels,
        alpha=0.5,
        label_smoothing=args.label_smoothing,
    )

    # FIX 2: Progressive unfreeze
    # Phase 1 (epochs 1–freeze_epochs): train head only (fast, stable)
    # Phase 2 (after freeze_epochs): unfreeze backbone for full fine-tune
    freeze_backbone(model)

    # Head-only optimizer with higher LR
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr * 5,          # higher LR when only head is training
        weight_decay=1e-2,
    )

    # FIX 3: CosineAnnealingWarmRestarts — resets LR every T_0 epochs
    # This breaks out of flat val acc plateaus (was the main issue in v1)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T0, T_mult=2, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    # --- Training ---
    best_val_acc  = 0.0
    patience_count = 0
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # FIX 2 (cont): unfreeze backbone at freeze_epochs + rebuild optimizer
        if epoch == args.freeze_epochs + 1:
            unfreeze_backbone(model)
            optimizer = AdamW(
                model.parameters(),
                lr=args.lr,              # normal LR for full fine-tune
                weight_decay=1e-2,
            )
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=args.T0, T_mult=2, eta_min=1e-6
            )
            logger.info("=== Switched to full fine-tune mode ===")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler
        )
        val_loss, val_acc, val_kappa = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(epoch)

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Kappa: {val_kappa:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
        )

        # KPI checks
        if val_acc >= 0.92:
            logger.info("🎉 KPI met: Accuracy ≥ 92%!")
        if val_kappa >= 0.85:
            logger.info("🎉 KPI met: Inter-rater Reliability (Kappa) ≥ 0.85!")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            save_model(
                model,
                str(output_dir / "best_model.pt"),
                metadata={"epoch": epoch, "val_acc": val_acc, "val_kappa": val_kappa},
            )
        else:
            patience_count += 1

        # Early stopping
        if patience_count >= args.patience:
            logger.info(f"Early stopping — no improvement for {args.patience} epochs.")
            break

    # --- Final Test Evaluation ---
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST EVALUATION")
    logger.info("=" * 60)
    test_loss, test_acc, test_kappa = evaluate(model, test_loader, loss_fn, device)
    logger.info(f"Test Accuracy : {test_acc:.4f}  (KPI target: ≥0.92)")
    logger.info(f"Test Kappa    : {test_kappa:.4f}  (KPI target: ≥0.85)")
    logger.info(f"Best val acc  : {best_val_acc:.4f}")
    logger.info(f"Best model    : {output_dir / 'best_model.pt'}")

    return test_acc, test_kappa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Detail Level Classifier v2")

    parser.add_argument("--data_dir",        default="data/fivek_384")
    parser.add_argument("--output_dir",      default="checkpoints_v3")
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--batch_size",      type=int,   default=32,
                        help="Reduced from 64→32 for 384px images (more VRAM per image)")
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--img_size",        type=int,   default=384)
    parser.add_argument("--num_levels",      type=int,   default=6)
    parser.add_argument("--num_workers",     type=int,   default=0)
    parser.add_argument("--synthetic_images",type=int,   default=500)
    # v2 specific
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing for CE loss (0=off, 0.1=recommended)")
    parser.add_argument("--freeze_epochs",   type=int,   default=8,
                        help="More freeze epochs for ConvNeXt — needs longer head warmup at 384px")
    parser.add_argument("--T0",              type=int,   default=10,
                        help="CosineAnnealingWarmRestarts restart period")
    parser.add_argument("--patience",        type=int,   default=15,
                        help="Early stopping patience epochs")

    args = parser.parse_args()
    main(args)
