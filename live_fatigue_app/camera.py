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
        
        # --- 1. Load Deep Learning Model (Safe Mode) ---
        self.model = self.load_model_safe(model_path)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # --- 2. Face Detector (Standard Haar Cascade) ---
        # We use the standard xml file included in cv2
        self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # --- 3. Smoothing Buffer ---
        self.prediction_buffer = deque(maxlen=10)
        
        # --- 4. Video Source ---
        self.source = source
        print(f"Opening source: {source}")
        self.video = cv2.VideoCapture(source)
        
        if not self.video.isOpened():
            self.valid_source = False
        else:
            self.valid_source = True

        # --- 5. Performance Vars ---
        self.frame_count = 0
        self.PROCESS_EVERY_N_FRAMES = 3  # Process 1 frame, skip 2 (Reduces lag)
        self.last_results = []           # Cache results to draw on skipped frames

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def load_model_safe(self, path):
        # Attempt NEW Architecture (Dropout)
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
            return model
        except:
            # Attempt OLD Architecture (Sigmoid)
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
                return model
            except Exception as e:
                print(f"Model Load Error: {e}")
                return model

    def create_error_frame(self, msg):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, msg, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

    def get_frame(self):
        if not self.valid_source:
            time.sleep(0.5)
            return self.create_error_frame(f"Cannot Open: {self.source}")

        success, frame = self.video.read()
        if not success:
            time.sleep(0.5)
            return self.create_error_frame("Stream Ended")

        self.frame_count += 1

        # --- LOGIC: Process every Nth frame to save CPU ---
        if self.frame_count % self.PROCESS_EVERY_N_FRAMES == 0:
            
            # 1. Resize for faster detection (work on small image)
            height, width = frame.shape[:2]
            target_width = 400
            ratio = target_width / float(width)
            new_height = int(height * ratio)
            
            frame_small = cv2.resize(frame, (target_width, new_height))
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            
            # 2. Enhance Contrast (Fixes detection for glasses/shadows)
            gray = cv2.equalizeHist(gray)
            
            # 3. Detect Faces
            haar_faces = self.haar_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(60, 60) # Ignore small speckles
            )
            
            current_results = []

            # --- FIX FOR "ALL OVER THE PLACE" ---
            # Logic: The driver is the largest face in the frame.
            # We calculate Area = w * h, sort descending, and take the top 1.
            if len(haar_faces) > 0:
                largest_face = sorted(haar_faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                
                # We only process this ONE face
                x_small, y_small, w_small, h_small = largest_face
                
                # Scale coords back up to original frame size
                x = int(x_small / ratio)
                y = int(y_small / ratio)
                w = int(w_small / ratio)
                h = int(h_small / ratio)

                # Clamp to frame
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)

                # 4. Run Deep Learning Model
                try:
                    face_img = frame[y:y+h, x:x+w]
                    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    pil_face = Image.fromarray(rgb_face)
                    input_tensor = self.transform(pil_face).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        swin_out = self.model(input_tensor)
                        prob = torch.sigmoid(swin_out).flatten().item()
                except:
                    prob = 0.0

                # 5. Smoothing
                self.prediction_buffer.append(prob)
                avg_prob = sum(self.prediction_buffer) / len(self.prediction_buffer)

                status = "FATIGUE" if avg_prob > 0.5 else "ACTIVE"
                color = (0, 0, 255) if avg_prob > 0.5 else (0, 255, 0)
                
                # Save for drawing
                current_results.append({
                    "box": (x, y, w, h),
                    "status": status,
                    "score": avg_prob,
                    "color": color
                })
            
            # Update cache
            self.last_results = current_results

        # --- DRAWING (Runs every frame from cache) ---
        for res in self.last_results:
            x, y, w, h = res["box"]
            color = res["color"]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(frame, f"{res['status']} ({res['score']:.0%})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Final resize for browser streaming efficiency
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (800, new_height))

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()