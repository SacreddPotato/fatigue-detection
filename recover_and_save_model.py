import torch
import os
# Import your model definition from the training file
from FatigueDetectionDemo import SwinT, swin_b_32

# --- CONFIGURATION ---
# REPLACE THIS with the actual path to your .ckpt file
CKPT_PATH = "saved_models/SwinTrans/Swin/lightning_logs/version_3/checkpoints/last.ckpt" 
OUTPUT_PATH = "saved_models/SwinTrans/swin_best.pth"

if not os.path.exists(CKPT_PATH):
    # Try to find it automatically if user didn't set it
    print(f"File not found: {CKPT_PATH}")
    print("Please edit CKPT_PATH in this script to point to your .ckpt file.")
    exit()

print(f"Loading checkpoint from {CKPT_PATH}...")
# Load the lightning checkpoint
checkpoint = torch.load(CKPT_PATH, map_location="cpu")

# Extract the state_dict
state_dict = checkpoint["state_dict"]

# Create a new dictionary for the standard PyTorch model
# Lightning adds "model." prefix to keys, we might need to remove it if your app expects pure Swin keys
# But your SwinT class wrapped it as self.model, so keys are likely "model.features..."
# Let's just save the model.model part to be safe for your app.

print("Extracting weights...")
# Re-instantiate the Lightning Module to load weights correctly
model = SwinT(swin_b_32)
model.load_state_dict(state_dict)

# Save the inner model weights (which your app expects)
torch.save(model.model.state_dict(), OUTPUT_PATH)

print(f"Success! Saved recovered weights to {OUTPUT_PATH}")
print("You can now run your app.")