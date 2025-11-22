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
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
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
        with open(self.annot_file, "r", encoding="utf-8") as f:
            for record in f.readlines():
                img_path, label = record.strip("\n").split()
                img_paths.append(img_path)
                labels.append(label)
        return img_paths, labels
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_paths[idx])
        
        img = Image.open(img_path.replace("\\", os.sep))
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        
        return img, label
    
    

#  Craft torch dataloader for UTA-RLDD
test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop((512, 512), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.Resize(224),
        transforms.ToTensor(),
        
    ]
)

# Use a named function for target_transform to avoid PicklingError
def to_int(x):
    return int(x)

train_set = UTARLDD(root_dir=DATASET_PATH, annot_file="train.txt", transform=train_transform, target_transform=to_int)
val_set = UTARLDD(root_dir=DATASET_PATH, annot_file="val.txt", transform=test_transform, target_transform=to_int)
test_set = UTARLDD(root_dir=DATASET_PATH, annot_file="test.txt", transform=test_transform, target_transform=to_int)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=2, drop_last=False)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=2, drop_last=False)


# Get labels directly without applying transforms
train_labels = [int(label) for label in train_set.labels]
val_labels = [int(label) for label in val_set.labels]

# Visualize some examples
# NUM_IMAGES = 4
# UTA_images = torch.stack([train_set[idx][0] for idx in range(NUM_IMAGES)], dim=0)
# img_grid = torchvision.utils.make_grid(UTA_images, nrow=NUM_IMAGES, normalize=True, pad_value=0.9)
# img_grid = img_grid.permute(1, 2, 0)

# plt.figure(figsize=(10, 10))
# plt.title("Image examples of the UTA-RLD dataset")
# plt.imshow(img_grid)
# plt.axis("off")
# plt.show()
# plt.close()

# Load the pretrained ViT (vit_b_32)
swin_b_32 = torchvision.models.swin_v2_s(weights=True)
swin_b_32 = swin_b_32
# summary(swin_b_32, (3, 1000, 1000))

# Customize the classification head
swin_b_32.head = nn.Sequential(
    nn.Linear(in_features=768, out_features=1, bias=True),
    nn.Sigmoid(),
)

# Make sure the entire model is trainable
for param in swin_b_32.parameters():
    param.requires_grad = True

# Test it with one image from UTA-RLDD
# img = train_set[0][0].repeat(1, 1, 1, 1)
# plt.imshow(img[0].permute(1, 2, 0))
# print(swin_b_32(img), "vs.", train_set[0][1])
# print(swin_b_32(img).shape)

class SwinT(L.LightningModule):
    def __init__(self, swin):
        super().__init__()
#         self.save_hyperparameters(ignore=["vit"])
        
        self.model = swin
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]
    
    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs).flatten()
        loss = F.binary_cross_entropy(preds, labels.to(torch.float32))
        f1 = binary_f1_score(preds, labels.to(torch.float32))

        self.log_dict({"%s_loss" % mode: loss, "%s_f1" % mode: f1}, prog_bar=True)
        return loss, f1

    def training_step(self, batch, batch_idx):
        loss, f1 = self._calculate_loss(batch, mode="train")
        # Debug: print labels and predictions for the first batch of each epoch
        if batch_idx == 0:
            imgs, labels = batch
            preds = self.model(imgs).flatten().detach().cpu().numpy()
            print(f"[DEBUG] Labels: {labels.detach().cpu().numpy()}")
            print(f"[DEBUG] Predictions: {preds}")
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
            ModelCheckpoint(mode="max", monitor="val_f1"),
            LearningRateMonitor("epoch"),
        ],
        log_every_n_steps=10,
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Always train from scratch
    L.seed_everything(42)  # To be reproducible
    model = l_vit
    trainer.fit(model, train_loader, val_loader)
    # Load best checkpoint after training
    model = SwinT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, swin=swin_b_32)

    # Test best model on validation and test set
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_f1"]}

    return model, result


# Main train loop
if __name__ == "__main__":
    model, result = train_fn(swin_model)
    print("Test F1 score:", result["test"])
    # Save the best model weights for later use
    torch.save(model.model.state_dict(), os.path.join(CHECKPOINT_PATH, "swin_best.pth"))
    
    
