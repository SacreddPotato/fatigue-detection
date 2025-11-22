import torch
import torchvision
import torch.nn as nn
from FatigueDetectionDemo import SwinT, CHECKPOINT_PATH, val_set, train_set
from torchvision import transforms
from PIL import Image
import random

# 1. Instantiate the Swin backbone
swin_b_32 = torchvision.models.swin_v2_s(weights=True)
swin_b_32.head = nn.Sequential(
    nn.Linear(in_features=768, out_features=1, bias=True),
    nn.Sigmoid(),
)

# 2. Create the LightningModule
model = SwinT(swin=swin_b_32)

# 3. Load the saved weights
model.model.load_state_dict(torch.load(f"{CHECKPOINT_PATH}/swin_best.pth", map_location="cpu"))
model.eval()

# Test on validation set samples
print("=== Testing on Validation Set ===")
for i in random.sample(range(len(val_set)), 10):
    img, label = val_set[i]
    with torch.no_grad():
        output = model(img.unsqueeze(0))
        prob = output.flatten().item()
        pred = int(prob > 0.5)
    print(f"Sample {i}: True label={label}, Predicted={pred}, Probability={prob:.4f}")

print("\n=== Testing on Training Set ===")
for i in random.sample(range(len(train_set)), 10):
    img, label = train_set[i]
    with torch.no_grad():
        output = model(img.unsqueeze(0))
        prob = output.flatten().item()
        pred = int(prob > 0.5)
    print(f"Sample {i}: True label={label}, Predicted={pred}, Probability={prob:.4f}")
