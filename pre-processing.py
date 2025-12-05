import os
import cv2
import dlib
from tqdm import tqdm

# CONFIG
ORIGINAL_DATASET_PATH = "./kaggle/input/uta-rldd" # Where your current images are
OUTPUT_DATASET_PATH = "./kaggle/input/uta-rldd-cropped" # Where to save cropped faces
ANNOTATION_FILES = ["train.txt", "val.txt", "test.txt"]

# Initialize Dlib (Same detector as your App)
detector = dlib.get_frontal_face_detector()

os.makedirs(OUTPUT_DATASET_PATH, exist_ok=True)

def process_file(filename):
    # Read the original text file
    input_path = os.path.join(ORIGINAL_DATASET_PATH, filename)
    output_path = os.path.join(OUTPUT_DATASET_PATH, filename)
    
    if not os.path.exists(input_path):
        print(f"Skipping {filename}, not found.")
        return

    new_lines = []
    
    with open(input_path, 'r') as f:
        lines = f.readlines()
        
    print(f"Processing {filename}...")
    for line in tqdm(lines):
        parts = line.strip().split()
        if len(parts) < 2: continue
        
        rel_path = parts[0]
        label = parts[1]
        
        # Load Image
        img_full_path = os.path.join(ORIGINAL_DATASET_PATH, rel_path)
        img = cv2.imread(img_full_path)
        
        if img is None: continue
        
        # Detect Face
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        
        if len(rects) > 0:
            # Take the largest face
            rect = max(rects, key=lambda r: r.area())
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            
            # Crop with some padding (optional, but good)
            face = img[max(0, y):min(img.shape[0], y+h), max(0, x):min(img.shape[1], x+w)]
            
            if face.size == 0: continue

            # Save Cropped Image
            save_path = os.path.join(OUTPUT_DATASET_PATH, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, face)
            
            # Write to new annotation file
            new_lines.append(f"{rel_path} {label}\n")
            
    with open(output_path, 'w') as f:
        f.writelines(new_lines)

for f in ANNOTATION_FILES:
    process_file(f)

print("Done! Use the new folder for training.")