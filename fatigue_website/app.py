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

# --- 1. Preprocessing (Single Definition) ---
# Ensure this matches the training exactly (Normalization included)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # Vital: This matches ImageNet statistics used in training
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# --- 2. Model Setup (Single Definition) ---
def get_model():
    model = torchvision.models.swin_v2_s(weights=None)
    model.head = nn.Sequential(
        nn.Dropout(p=0.5),  # Kept to match the state_dict keys
        nn.Linear(in_features=768, out_features=1, bias=True),
        # nn.Sigmoid() removed here because we use BCEWithLogitsLoss in training
    )
    return model

# --- Load Model ---
print(f"Loading model from {MODEL_PATH}...")
model = get_model()

if os.path.exists(MODEL_PATH):
    # map_location ensures it loads on CPU if CUDA is not available
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")
else:
    print("WARNING: Model file not found. Predictions will be random.")

model.to(DEVICE)
model.eval()

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    error_msg = None
    image_data = None

    if request.method == 'POST':
        # 1. Validation: Check if file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded.")
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error="No file selected.")

        if file:
            try:
                # 2. Open the image
                img = Image.open(file.stream).convert('RGB')
                
                # 3. Convert to Base64 for display (Optional but nice for UI)
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                image_data = f"data:image/jpeg;base64,{img_str}"
                
                # 4. Transform image for model
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                
                # 5. Run Inference
                with torch.no_grad():
                    output = model(img_tensor)
                    
                    # Apply Sigmoid here because we removed it from the model
                    prob = torch.sigmoid(output).flatten().item() 
                    
                    pred_class = int(prob > 0.5)
                
                # 6. Decode Result
                label = "Fatigue Detected" if pred_class == 1 else "Active / Alert"
                # Confidence calculation: if prob is 0.1 (Active), confidence is 0.9 (90% sure it's Active)
                confidence = prob if pred_class == 1 else 1 - prob
                
                prediction_result = {
                    "label": label,
                    "probability": f"{confidence:.2%}",
                    "raw_score": f"{prob:.4f}"
                }

            except Exception as e:
                error_msg = f"Error processing image: {str(e)}"

    # Return template with all variables
    return render_template('index.html', result=prediction_result, image_data=image_data, error=error_msg)

if __name__ == '__main__':
    app.run(debug=True)