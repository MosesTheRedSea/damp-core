import torch
from pathlib import Path

from src.damp import DampNet

# -------------------------------------------------
# CHANGE THESE
# -------------------------------------------------

CHECKPOINT_PATH = Path("/home/3/um07293/research/occul-net/models/run_3/damp_best.pth")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_OBJECT_CLASSES = 25
NUM_MATERIAL_CLASSES = 6

# -------------------------------------------------
# Load model
# -------------------------------------------------

model = DampNet(
    NUM_OBJECT_CLASSES,
    NUM_MATERIAL_CLASSES
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)
model.eval()

print("=" * 60)
print("MODEL LOADED")
print("=" * 60)

# -------------------------------------------------
# Method 1
# -------------------------------------------------

print("\nChecking detection head...")

if hasattr(model, "det_head"):
    print(f"det_head.out_features = {model.det_head.out_features}")
else:
    print("Model has no attribute named 'det_head'.")

# -------------------------------------------------
# Method 2
# -------------------------------------------------

print("\nFirst parameter shape:")

try:
    print(next(model.parameters()).shape)
except Exception as e:
    print(e)

# -------------------------------------------------
# Method 3
# -------------------------------------------------

print("\nRunning dummy forward pass...")

dummy_ir = torch.randn(
    1,
    16,
    512
).to(DEVICE)

dummy_spec = torch.randn(
    1,
    16,
    129,
    3
).to(DEVICE)

with torch.no_grad():
    det, dist, mat = model(dummy_ir, dummy_spec)

print("\nOutput Shapes")
print("------------------------")
print("Detection :", det.shape)
print("Distance  :", dist.shape)
print("Material  :", mat.shape)

print("\nDetection classes:", det.shape[1])

print("=" * 60)

if det.shape[1] == 25:
    print("✓ Model was trained with 25 object classes.")
    print("✓ Do NOT retrain.")
    print("✓ Fix your OBJ_CLASSES list (it currently has 26 names).")

elif det.shape[1] == 26:
    print("⚠ Model was trained with 26 outputs.")
    print("Check your training code and dataset before changing OBJ_CLASSES.")

else:
    print(f"Unexpected number of detection outputs: {det.shape[1]}")

print("=" * 60)