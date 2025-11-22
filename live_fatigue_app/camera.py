import cv2
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import deque
import time

class VideoCamera(object):
    def __init__(self, model_path='swin_best.pth', source=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Safe Model Loading
        self.model = self.load_model_safe(model_path)
        
        # 2. Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 3. Face Detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.prediction_buffer = deque(maxlen=10)
        
        # 4. Open Video Source
        self.source = source
        print(f"Attempting to open source: {source}")
        self.video = cv2.VideoCapture(source)
        
        # Check validity immediately
        if not self.video.isOpened():
            self.valid_source = False
            print(f"ERROR: Could not open source: {source}")
        else:
            self.valid_source = True

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def load_model_safe(self, path):
        print(f"Loading Live Model from {path}...")
        
        # Attempt 1: NEW Architecture
        model = torchvision.models.swin_v2_s(weights=None)
        model.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=768, out_features=1, bias=True),
        )
        try:
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            print("SUCCESS: Loaded New Model Architecture.")
            return model
        except RuntimeError:
            pass

        # Attempt 2: OLD Architecture
        model = torchvision.models.swin_v2_s(weights=None)
        model.head = nn.Sequential(
            nn.Linear(in_features=768, out_features=1, bias=True),
            nn.Sigmoid(), 
        )
        try:
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            print("SUCCESS: Loaded Old Model Architecture.")
            return model
        except Exception as e:
            print(f"FATAL ERROR: Could not load model. {e}")
            return model

    def create_error_frame(self, message):
        # Creates a black image with red text
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add timestamp to show it's not frozen
        timestamp = time.strftime("%H:%M:%S")
        full_message = f"{message} ({timestamp})"
        
        cv2.putText(img, full_message, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

    def get_frame(self):
        # --- FIX FOR LAG: Throttle errors ---
        if not self.valid_source:
            time.sleep(0.5) # Sleep to prevent CPU spike
            return self.create_error_frame(f"Cannot Open: {self.source}")

        success, frame = self.video.read()
        if not success:
            time.sleep(0.5) # Sleep to prevent CPU spike
            return self.create_error_frame("Stream Ended / Connection Failed")

        # 1. Enhanced Face Detection (Fix for Glasses/Lighting)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray) 
        
        faces = self.face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))

        status_text = "No Face Detected"
        color = (255, 0, 0) # Blue

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            
            try:
                # Preprocess
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                pil_face = Image.fromarray(rgb_face)
                input_tensor = self.transform(pil_face).unsqueeze(0).to(self.device)

                # Inference
                with torch.no_grad():
                    output = self.model(input_tensor)
                    prob = torch.sigmoid(output).flatten().item()

                # Smoothing
                self.prediction_buffer.append(prob)
                avg_prob = sum(self.prediction_buffer) / len(self.prediction_buffer)
                
                is_fatigued = avg_prob > 0.5
                
                if is_fatigued:
                    status_text = f"FATIGUE ({avg_prob:.0%})"
                    color = (0, 0, 255)
                else:
                    status_text = f"ACTIVE ({1-avg_prob:.0%})"
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
            except Exception:
                pass

        cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Optimization: Resize large frames
        height, width = frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (1280, new_height))

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()