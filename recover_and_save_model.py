import torch
import torchvision
import torch.nn as nn
from FatigueDetectionDemo import SwinT

# Path to your checkpoint
ckpt_path = "saved_models/SwinTrans/Siwn/lightning_logs/version_3/checkpoints/epoch=1-step=1596.ckpt"
# Path to save the recovered weights
save_path = "saved_models/SwinTrans/swin_best.pth"

# 1. Instantiate your Swin backbone
swin_b_32 = torchvision.models.swin_v2_s(weights=True)
swin_b_32.head = nn.Sequential(
    nn.Linear(in_features=768, out_features=1, bias=True),
    nn.Sigmoid(),
)

# 2. Create your LightningModule
model = SwinT(swin=swin_b_32)

# 3. Load the checkpoint
checkpoint = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])

# 4. Save the model weights
torch.save(model.model.state_dict(), save_path)
print(f"Recovered model weights saved to {save_path}")
