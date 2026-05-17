# -*- coding: utf-8 -*-
"""ONNX export + latency benchmark for ConvNeXt-Small classifier."""
import sys, os, warnings
sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

import torch
import time
import numpy as np

# Suppress torch.onnx verbose Unicode output on Windows
import io
_old_stdout = sys.stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from model.classifier import DetailLevelClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load best model
checkpoint = torch.load("checkpoints_v3/best_model.pt", map_location=device)
model = DetailLevelClassifier(num_classes=6, pretrained=False).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Model loaded.")

dummy = torch.randn(1, 3, 224, 224).to(device)

# Export ONNX
with torch.no_grad():
    torch.onnx.export(
        model.cpu(),
        dummy.cpu(),
        "checkpoints_v3/model.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        verbose=False,
    )
model.to(device)
print("ONNX exported: checkpoints_v3/model.onnx")

# GPU latency benchmark
times = []
with torch.no_grad():
    for _ in range(100):
        t = time.perf_counter()
        model(dummy)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
avg_gpu = sum(times) / len(times)
print(f"PyTorch GPU latency: {avg_gpu:.1f}ms avg  |  {max(times):.1f}ms max")

# ONNX CPU latency
import onnxruntime as ort
sess = ort.InferenceSession(
    "checkpoints_v3/model.onnx",
    providers=["CPUExecutionProvider"]
)
inp = dummy.cpu().numpy()
times2 = []
for _ in range(30):
    t = time.perf_counter()
    sess.run(None, {"image": inp})
    times2.append((time.perf_counter() - t) * 1000)
avg_cpu = sum(times2) / len(times2)
print(f"ONNX CPU latency:    {avg_cpu:.1f}ms avg  |  {max(times2):.1f}ms max")

print()
print("=" * 40)
print("LATENCY KPI (<=1500ms):", "PASS" if avg_cpu < 1500 else "FAIL")
print("=" * 40)
