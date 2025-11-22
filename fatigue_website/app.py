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

# --- Model Setup ---
def get_model():
    model = torchvision.models.swin_v2_s(weights=None)
    model.head = nn.Sequential(
        nn.Linear(in_features=768, out_features=1, bias=True),
        nn.Sigmoid(),
    )
    return model

# Load Model
print(f"Loading model from {MODEL_PATH}...")
model = get_model()

if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")
else:
    print("WARNING: Model file not found. Predictions will be random.")

model.to(DEVICE)
model.eval()

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

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
                
                # --- NEW: Convert image to Base64 for display ---
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                image_data = f"data:image/jpeg;base64,{img_str}"
                # -----------------------------------------------
                
                # 2. Transform image for model
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                
                # 3. Run Inference
                with torch.no_grad():
                    output = model(img_tensor)
                    prob = output.flatten().item()
                    pred_class = int(prob > 0.5)
                
                # 4. Decode Result
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