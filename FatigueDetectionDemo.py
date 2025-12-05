import os
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional.classification import binary_f1_score, binary_accuracy
from torchvision import transforms
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

# --- CONFIGURATION ---
# ENSURE THIS POINTS TO YOUR NEW CROPPED DATASET
DATASET_PATH = os.environ.get("PATH_DATASETS", "./kaggle/input/uta-rldd-cropped")
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/SwinTrans/")
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Ensure reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

# --- 1. DATASET DEFINITION ---
class UTARLDD(Dataset):
    def __init__(self, root_dir, annot_file, transform=None):
        self.root_dir = root_dir
        self.annot_file = os.path.join(root_dir, annot_file)
        self.transform = transform
        self.img_paths, self.labels = self.read_annot()
    
    def read_annot(self):
        img_paths = []
        labels = []
        # Safe reading of the text file
        if not os.path.exists(self.annot_file):
            print(f"[WARNING] Annotation file not found: {self.annot_file}")
            return [], []
            
        with open(self.annot_file, "r", encoding="utf-8") as f:
            for record in f.readlines():
                parts = record.strip().split()
                if len(parts) >= 2:
                    img_paths.append(parts[0])
                    # TRIPLE CHECK: Directly parse binary labels (0 or 1) as float
                    labels.append(float(parts[1]))
        return img_paths, labels
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_paths[idx])
        try:
            img = Image.open(img_path.replace("\\", os.sep)).convert('RGB')
        except Exception as e:
            print(f"[WARN] Corrupt image at {img_path}: {e}")
            img = Image.new('RGB', (224, 224))
            
        label = self.labels[idx]
        if self.transform: 
            img = self.transform(img)
        
        # Return label as float32 tensor for BCE Loss
        return img, torch.tensor(label, dtype=torch.float32)

# --- 2. TRANSFORMS (Optimized for Cropped Faces) ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])

# Setup DataLoaders
# Set num_workers=0 for Windows compatibility to avoid BrokenPipeError
train_set = UTARLDD(root_dir=DATASET_PATH, annot_file="train.txt", transform=train_transform)
val_set = UTARLDD(root_dir=DATASET_PATH, annot_file="val.txt", transform=test_transform)
test_set = UTARLDD(root_dir=DATASET_PATH, annot_file="test.txt", transform=test_transform)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)

# --- 3. LIGHTNING MODULE ---
class SwinT(L.LightningModule):
    def __init__(self):
        super().__init__()
        # Load Pretrained Swin
        self.model = torchvision.models.swin_v2_s(weights='DEFAULT')
        
        # Replace Head
        self.model.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=768, out_features=1, bias=True)
        )
        
        # Freezing Logic: Start with frozen backbone
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        # Start with a conservative learning rate
        optimizer = optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        
        # Safe Scheduler: Only lowers LR if loss plateaus. No spikes.
        # FIXED: Removed 'verbose=True' which crashes on newer PyTorch versions
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3
        )
        
        # TRIPLE CHECK: Correct nesting for Lightning to read validation loss
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self):
        # Unfreeze backbone at epoch 5
        if self.current_epoch == 5:
            print("\n>>> UNFREEZING BACKBONE (Fine-tuning enabled) <<<\n")
            for param in self.model.parameters():
                param.requires_grad = True

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        logits = self.model(imgs).flatten()
        
        # Weighted Loss to handle class imbalance (1.5x for Fatigue)
        pos_weight = torch.tensor([1.5], device=self.device)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        
        probs = torch.sigmoid(logits)
        acc = binary_accuracy(probs, labels)
        f1 = binary_f1_score(probs, labels)

        self.log_dict({f"{mode}_loss": loss, f"{mode}_acc": acc, f"{mode}_f1": f1}, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

# --- 4. TRAINING EXECUTION ---
if __name__ == "__main__":
    swin_model = SwinT()
    
    # Optional: Clean start logic
    # If you want to force a clean start, delete 'swin_best.pth' before running
    final_weight_path = os.path.join(CHECKPOINT_PATH, "swin_best.pth")
    if os.path.exists(final_weight_path):
        print(f"[INFO] Found previous weights at {final_weight_path}. Loading...")
        try:
            swin_model.model.load_state_dict(torch.load(final_weight_path), strict=False)
            print("[SUCCESS] Weights loaded.")
        except Exception as e:
            print(f"[WARN] Could not load weights: {e}. Starting from scratch.")

    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "Swin_Logs"),
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        max_epochs=20, # 20 is perfect for the new cropped dataset
        callbacks=[
            # Save top 3 models
            ModelCheckpoint(mode="max", monitor="val_f1", save_top_k=3, filename="swin-{epoch:02d}-{val_f1:.2f}"),
            LearningRateMonitor("epoch"),
            EarlyStopping(monitor="val_f1", patience=5, mode="max")
        ],
        log_every_n_steps=10
    )

    trainer.fit(swin_model, train_loader, val_loader)
    
    # Automatically load the best checkpoint found during training
    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        print(f"Loading best checkpoint from: {best_path}")
        swin_model = SwinT.load_from_checkpoint(best_path)
    
    trainer.test(swin_model, dataloaders=test_loader)
    
    # Final Save for App
    torch.save(swin_model.model.state_dict(), final_weight_path)
    print(f"Training Complete! Optimized model saved to: {final_weight_path}")