import os

DATASET_PATH = "./kaggle/input/uta-rldd"

# Check train.txt
print("=== Checking train.txt ===")
with open(os.path.join(DATASET_PATH, "train.txt"), "r") as f:
    train_lines = f.readlines()[:10]
    for line in train_lines:
        print(line.strip())

print(f"\nTotal training samples: {len(open(os.path.join(DATASET_PATH, 'train.txt')).readlines())}")

# Count class distribution
train_labels = []
with open(os.path.join(DATASET_PATH, "train.txt"), "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            train_labels.append(int(parts[1]))

val_labels = []
with open(os.path.join(DATASET_PATH, "val.txt"), "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            val_labels.append(int(parts[1]))

print(f"\nTraining class distribution:")
print(f"  Class 0 (active): {train_labels.count(0)}")
print(f"  Class 1 (fatigue): {train_labels.count(1)}")

print(f"\nValidation class distribution:")
print(f"  Class 0 (active): {val_labels.count(0)}")
print(f"  Class 1 (fatigue): {val_labels.count(1)}")
