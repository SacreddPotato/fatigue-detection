import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from flask import Flask, request, render_template
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = "swin_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Preprocessing ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# --- 2. Robust Model Loader ---
def load_model_safe(path, device):
    print(f"Attempting to load model from {path}...")
    
    if not os.path.exists(path):
        print("WARNING: Model file not found. Predictions will be random.")
        # Return a random initialized model just so app doesn't crash
        model = torchvision.models.swin_v2_s(weights=None)
        model.head = nn.Sequential(nn.Linear(768, 1))
        return model.to(device)

    # Attempt 1: Try New Architecture (Dropout -> Linear)
    # This is what you will have AFTER you retrain
    model = torchvision.models.swin_v2_s(weights=None)
    model.head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=768, out_features=1, bias=True)
    )
    
    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("SUCCESS: Loaded New Model Architecture (Dropout Enabled).")
        return model
    except RuntimeError:
        print("Warning: New architecture match failed. Trying old architecture...")

    # Attempt 2: Try Old Architecture (Linear -> Sigmoid)
    # This is what you have NOW
    model = torchvision.models.swin_v2_s(weights=None)
    model.head = nn.Sequential(
        nn.Linear(in_features=768, out_features=1, bias=True),
        nn.Sigmoid()
    )
    
    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        
        # CRITICAL FIX:
        # The old model outputs Probabilities (0-1) because of Sigmoid.
        # The new model outputs Logits (-inf to +inf).
        # Your inference code expects Logits (because it calls torch.sigmoid).
        # So, we remove the Sigmoid layer from this old model after loading weights.
        model.head = nn.Sequential(
            model.head[0] # Keep only the Linear layer
        )
        
        model.to(device)
        model.eval()
        print("SUCCESS: Loaded Old Model Architecture (Fixed for compatibility).")
        return model
    except Exception as e:
        print(f"FATAL ERROR: Could not load model. {e}")
        return None

# --- Load Model ---
model = load_model_safe(MODEL_PATH, DEVICE)

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    error_msg = None
    image_data = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded.")
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error="No file selected.")

        if file:
            try:
                # 1. Open the image
                img = Image.open(file.stream).convert('RGB')
                
                # 2. Convert to Base64 for display
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                image_data = f"data:image/jpeg;base64,{img_str}"
                
                # 3. Transform image
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                
                # 4. Run Inference
                with torch.no_grad():
                    output = model(img_tensor)
                    
                    # Always apply sigmoid now, because load_model_safe ensures 
                    # we output logits regardless of which model file is used.
                    prob = torch.sigmoid(output).flatten().item() 
                    
                    pred_class = int(prob > 0.5)
                
                # 5. Decode Result
                label = "Fatigue Detected" if pred_class == 1 else "Active / Alert"
                confidence = prob if pred_class == 1 else 1 - prob
                
                prediction_result = {
                    "label": label,
                    "probability": f"{confidence:.2%}",
                    "raw_score": f"{prob:.4f}"
                }

            except Exception as e:
                error_msg = f"Error processing image: {str(e)}"

    return render_template('index.html', result=prediction_result, image_data=image_data, error=error_msg)

if __name__ == '__main__':
    app.run(debug=True)