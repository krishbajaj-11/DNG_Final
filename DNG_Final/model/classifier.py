"""
Multi-Level Detail Enhancement Classifier
ConvNeXt-Small backbone with ordinal classification head
Samsung Multi-Level Detail Enhancement Detection Project

v3: Upgraded from EfficientNet-B3 → ConvNeXt-Small
    ConvNeXt uses large-kernel depthwise convolutions that capture
    fine-grained texture differences across spatial scales — ideal
    for distinguishing subtle sharpness levels in enhancement-based datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Dict


class DetailLevelClassifier(nn.Module):
    """
    ConvNeXt-Small backbone fine-tuned for 6-class ordinal classification
    of RAW image detail enhancement levels (L5 = baseline, L10 = max enhanced).

    Architecture:
        ConvNeXt-Small feature extractor  (pretrained on ImageNet-1K)
        → Adaptive Average Pooling → Flatten
        → LayerNorm(768)
        → Dropout(0.3)
        → FC(768 → 384) + GELU + LayerNorm
        → Dropout(0.2)
        → FC(384 → num_classes)

    Why ConvNeXt over EfficientNet for this task:
        - Large 7×7 depthwise kernels → better at capturing texture gradients
        - LayerNorm instead of BatchNorm → more stable on small feature shifts
        - Stronger fine-grained texture discrimination (proven on FGVC benchmarks)
    """

    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained ConvNeXt-Small
        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.convnext_small(weights=weights)

        # Feature extractor (all stages) + pooling
        self.features = backbone.features          # ConvNeXt stages
        self.avgpool  = backbone.avgpool           # AdaptiveAvgPool2d

        in_features = 768  # ConvNeXt-Small output channels

        # Classification head — uses GELU + LayerNorm (ConvNeXt style)
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Dropout(p=0.2),
            nn.Linear(384, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) normalized image tensor (224 or 384)
        Returns:
            logits: (B, num_classes)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (predicted_class_0indexed, confidence_score).
        Add 5 to get human-readable level (L5–L10).
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        predicted = probs.argmax(dim=1)
        confidence = probs.max(dim=1).values
        return predicted, confidence

    def get_level_probabilities(self, x: torch.Tensor) -> Dict[int, float]:
        """Return probability for each level as a dict {5: p5, 6: p6, ...}."""
        logits = self.forward(x.unsqueeze(0) if x.dim() == 3 else x)
        probs = F.softmax(logits, dim=1)[0]
        return {i + 5: float(p) for i, p in enumerate(probs)}


# ------------------------------------------------------------------
# Ordinal Loss — better than plain CrossEntropy for ordered classes
# ------------------------------------------------------------------

class OrdinalCrossEntropyLoss(nn.Module):
    """
    Combines standard CrossEntropy with a penalty term for predictions
    that are far from the true label (ordinal mismatch penalty).
    This encourages the model to predict nearby levels rather than
    wildly incorrect ones.
    """

    def __init__(self, num_classes: int = 10, alpha: float = 0.5,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        # label_smoothing: stops over-penalising near-miss level confusion
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross-entropy with optional label smoothing
        ce_loss = self.ce(logits, targets)

        # Ordinal penalty: penalize proportional to distance from true label
        probs = F.softmax(logits, dim=1)                          # (B, C)
        levels = torch.arange(self.num_classes, device=logits.device).float()  # (C,)
        expected = (probs * levels).sum(dim=1)                    # (B,)
        ordinal_loss = F.mse_loss(expected, targets.float())

        return ce_loss + self.alpha * ordinal_loss


# ------------------------------------------------------------------
# Model factory and utilities
# ------------------------------------------------------------------

def build_model(num_classes: int = 10, pretrained: bool = True) -> DetailLevelClassifier:
    """Convenience factory to build the model."""
    model = DetailLevelClassifier(num_classes=num_classes, pretrained=pretrained)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    return model


def save_model(model: DetailLevelClassifier, path: str, metadata: dict = None):
    """Save model weights + optional metadata."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_classes": model.num_classes,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(path: str, device: str = "cpu") -> DetailLevelClassifier:
    """Load a saved model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model = DetailLevelClassifier(num_classes=checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    model = build_model(num_classes=10)
    model.eval()

    dummy = torch.randn(4, 3, 224, 224)
    logits = model(dummy)
    print(f"Logits shape: {logits.shape}")

    pred, conf = model.predict(dummy)
    print(f"Predicted levels (0-idx): {pred.tolist()}")
    print(f"Confidence scores: {[round(c.item(), 3) for c in conf]}")

    loss_fn = OrdinalCrossEntropyLoss()
    targets = torch.tensor([3, 7, 1, 9])
    loss = loss_fn(logits, targets)
    print(f"Ordinal loss: {loss.item():.4f}")
