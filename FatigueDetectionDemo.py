# Common import
import os
import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional.classification import binary_f1_score
from torchsummary import summary
from torchvision import transforms
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


# Path to the folder where the datasets are/should be downloaded
DATASET_PATH = os.environ.get("PATH_DATASETS", "./kaggle/input/uta-rldd")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/SwinTrans/")
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# Build torch dataset for UTA-RLDD
class UTARLDD(Dataset):
    def __init__(
        self, 
        root_dir, 
        annot_file,
        transform=None,
        target_transform=None,
    ):
        self.root_dir = root_dir
        self.annot_file = os.path.join(root_dir, annot_file)
        self.transform = transform
        self.target_transform = target_transform
        
        self.img_paths, self.labels = self.read_annot()
    
    def read_annot(self):
        img_paths = []
        labels = []
        # Using utf-8 to handle any potential character encoding issues
        with open(self.annot_file, "r", encoding="utf-8") as f:
            for record in f.readlines():
                parts = record.strip("\n").split()
                if len(parts) >= 2:
                    img_path = parts[0]
                    label = parts[1]
                    img_paths.append(img_path)
                    labels.append(label)
        return img_paths, labels
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_paths[idx])
        
        # Handle path separators for different OS
        img = Image.open(img_path.replace("\\", os.sep)).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        
        return img, label
    
    

#  Craft torch dataloader for UTA-RLDD

# --- FIX 1: Added Normalization for Pretrained Models ---
# ImageNet stats (standard for Swin/ResNet/ViT)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])

test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize, # <--- Critical for correct colors/contrast perception
    ]
)

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        normalize, # <--- Critical for correct colors/contrast perception
    ]
)

# Use a named function for target_transform to avoid PicklingError
def to_int(x):
    return int(x)

train_set = UTARLDD(root_dir=DATASET_PATH, annot_file="train.txt", transform=train_transform, target_transform=to_int)
val_set = UTARLDD(root_dir=DATASET_PATH, annot_file="val.txt", transform=test_transform, target_transform=to_int)
test_set = UTARLDD(root_dir=DATASET_PATH, annot_file="test.txt", transform=test_transform, target_transform=to_int)

# Reduced num_workers to 0 or 1 for Windows stability, increased for Linux/Mac
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=2, drop_last=False)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=2, drop_last=False)


# Load the pretrained Swin Transformer
swin_b_32 = torchvision.models.swin_v2_s(weights=True)

# --- FIX 2: Remove Sigmoid from Architecture & Add Dropout ---
# We remove Sigmoid here to use BCEWithLogitsLoss later (better stability).
# We add Dropout to prevent the "99% confidence" overfitting issue.
swin_b_32.head = nn.Sequential(
    nn.Dropout(p=0.5), # <--- Prevents overconfidence
    nn.Linear(in_features=768, out_features=1, bias=True)
    # No Sigmoid here!
)

# Make sure the entire model is trainable
for param in swin_b_32.parameters():
    param.requires_grad = True


class SwinT(L.LightningModule):
    def __init__(self, swin):
        super().__init__()
        self.model = swin
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]
    
    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        
        # Preds are now "logits" (raw scores from -inf to +inf)
        logits = self.model(imgs).flatten()
        
        # --- FIX 3: Use BCEWithLogitsLoss ---
        # This combines Sigmoid + BCELoss in a numerically stable way
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float32))
        
        # Calculate probabilities manually for F1 score and logging
        probs = torch.sigmoid(logits)
        f1 = binary_f1_score(probs, labels.to(torch.float32))

        self.log_dict({"%s_loss" % mode: loss, "%s_f1" % mode: f1}, prog_bar=True)
        return loss, f1, probs

    def training_step(self, batch, batch_idx):
        loss, f1, probs = self._calculate_loss(batch, mode="train")
        
        # Debug: print labels and predictions for the first batch of each epoch
        if batch_idx == 0:
            imgs, labels = batch
            # We use the calculated probs from above
            preds_numpy = probs.detach().cpu().numpy()
            print(f"\n[DEBUG] Labels: {labels.detach().cpu().numpy()}")
            print(f"[DEBUG] Probabilities: {preds_numpy}")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
        
swin_model = SwinT(swin=swin_b_32)

# Setup train loop
def train_fn(l_vit):
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "Siwn"),
        accelerator="auto",
        devices=1,
        max_epochs=150,
        gradient_clip_val=1,
        callbacks=[
            ModelCheckpoint(mode="max", monitor="val_f1", save_last=True),
            LearningRateMonitor("epoch"),
            # --- FIX 4: Early Stopping ---
            # Stops training if validation F1 score doesn't improve for 10 epochs
            EarlyStopping(monitor="val_f1", patience=10, mode="max")
        ],
        log_every_n_steps=10,
    )
    trainer.logger._log_graph = False 
    trainer.logger._default_hp_metric = None 

    L.seed_everything(42)
    model = l_vit
    trainer.fit(model, train_loader, val_loader)
    
    # Load best checkpoint
    # Note: We must pass the architecture (swin=swin_b_32) when loading
    if trainer.checkpoint_callback.best_model_path:
        print(f"Loading best model from: {trainer.checkpoint_callback.best_model_path}")
        model = SwinT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, swin=swin_b_32)
    
    # Test best model
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_f1"]}

    return model, result


# Main train loop
if __name__ == "__main__":
    model, result = train_fn(swin_model)
    print("Test F1 score:", result["test"])
    # Save the best model weights
    torch.save(model.model.state_dict(), os.path.join(CHECKPOINT_PATH, "swin_best.pth"))