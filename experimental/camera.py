import cv2
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import deque
import time
import os
from scipy.spatial import distance as dist

# --- DLIB IMPORT ---
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    print("WARNING: Dlib not found. Please activate your 'fatigue_env' environment.")
    DLIB_AVAILABLE = False

class VideoCamera(object):
    def __init__(self, model_path='swin_best.pth', predictor_path='shape_predictor_68_face_landmarks.dat', source=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model_safe(model_path)
        self.transform = transforms.Compose([
        transforms.Resize((224, 224)), # Force exact squeeze, NO CROP
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

        self.dlib_ready = False
        if DLIB_AVAILABLE and os.path.exists(predictor_path):
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(predictor_path)
            self.dlib_ready = True
        
        self.prediction_buffer = deque(maxlen=30)
        self.EYE_AR_THRESH = 0.25
        self.MAR_THRESH = 0.6 

        self.source = source
        self.video = cv2.VideoCapture(source)
        self.valid_source = self.video.isOpened()

        self.frame_count = 0
        self.PROCESS_EVERY_N_FRAMES = 3 
        self.last_results = []           

    def __del__(self):
        if self.video.isOpened(): self.video.release()

    def load_model_safe(self, path):
        model = torchvision.models.swin_v2_s(weights=None)
        model.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(768, 1))
        try:
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            return model
        except:
            model = torchvision.models.swin_v2_s(weights=None)
            model.head = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())
            try:
                state_dict = torch.load(path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device).eval()
                return model
            except: return model

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def mouth_aspect_ratio(self, mouth):
        A = dist.euclidean(mouth[2], mouth[10]) 
        B = dist.euclidean(mouth[4], mouth[8])
        C = dist.euclidean(mouth[0], mouth[6]) 
        return (A + B) / (2.0 * C)

    def create_error_frame(self, msg):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, msg, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

    def draw_dashboard(self, frame, metrics):
        """Draws a breakdown of the calculation on the screen."""
        overlay = frame.copy()
        
        # Dashboard dimensions
        box_h = 180
        box_w = 300
        padding = 10
        
        # Draw semi-transparent background
        cv2.rectangle(overlay, (padding, padding), (padding + box_w, padding + box_h), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        white = (255, 255, 255)
        green = (0, 255, 0)
        red = (0, 0, 255)
        yellow = (0, 255, 255)
        
        # 1. Swin Score (50%)
        swin_val = metrics['swin']
        swin_color = red if swin_val > 0.5 else green
        cv2.putText(frame, f"Swin Model (50%):  {swin_val:.2f}", (20, 40), font, scale, white, 1)
        # Visual Bar
        cv2.rectangle(frame, (180, 30), (180 + int(swin_val * 100), 40), swin_color, -1)

        # 2. EAR Score (30%)
        if self.dlib_ready:
            ear_prob = metrics['ear_prob'] # The calculated probability (0-1), not raw EAR
            ear_color = red if ear_prob > 0.5 else green
            cv2.putText(frame, f"Eyes/EAR (30%):    {ear_prob:.2f}", (20, 70), font, scale, white, 1)
            cv2.rectangle(frame, (180, 60), (180 + int(ear_prob * 100), 70), ear_color, -1)
            # Show raw value small
            cv2.putText(frame, f"(Raw: {metrics['ear_val']:.2f})", (20, 85), font, 0.4, (200,200,200), 1)

            # 3. MAR Score (20%)
            mar_prob = metrics['mar_prob']
            mar_color = red if mar_prob > 0.5 else green
            cv2.putText(frame, f"Mouth/MAR (20%):   {mar_prob:.2f}", (20, 110), font, scale, white, 1)
            cv2.rectangle(frame, (180, 100), (180 + int(mar_prob * 100), 110), mar_color, -1)
            cv2.putText(frame, f"(Raw: {metrics['mar_val']:.2f})", (20, 125), font, 0.4, (200,200,200), 1)
        else:
            cv2.putText(frame, "Geometry: N/A", (20, 70), font, scale, (100,100,100), 1)

        # 4. Final Sum
        final = metrics['score']
        status = "FATIGUE" if final > 0.5 else "ACTIVE"
        status_color = red if final > 0.5 else green
        
        cv2.line(frame, (20, 140), (280, 140), white, 1)
        cv2.putText(frame, f"TOTAL: {final:.2f} [{status}]", (20, 165), font, 0.6, status_color, 2)

    def get_frame(self):
        if not self.valid_source:
            time.sleep(0.5)
            return self.create_error_frame(f"Cannot Open: {self.source}")

        success, frame = self.video.read()
        if not success:
            time.sleep(0.5)
            return self.create_error_frame("Stream Ended")

        self.frame_count += 1

        if self.frame_count % self.PROCESS_EVERY_N_FRAMES == 0:
            height, width = frame.shape[:2]
            target_width = 400
            ratio = target_width / float(width)
            new_height = int(height * ratio)
            frame_small = cv2.resize(frame, (target_width, new_height))
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            
            current_results = []
            rects = []

            if self.dlib_ready:
                rects = self.detector(gray, 0)
                if len(rects) > 0:
                    rects = [max(rects, key=lambda r: r.area())]
            
            # Default metrics if no face found
            if not rects:
                 self.last_results = [{
                    "box": None,
                    "swin": 0.0, "ear_val": 0.0, "ear_prob": 0.0, 
                    "mar_val": 0.0, "mar_prob": 0.0, "score": 0.0,
                    "landmarks": [], "ratio": ratio
                }]

            for rect in rects:
                x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                x_orig = int(x / ratio); y_orig = int(y / ratio)
                w_orig = int(w / ratio); h_orig = int(h / ratio)
                x_orig = max(0, x_orig); y_orig = max(0, y_orig)
                w_orig = min(w_orig, width - x_orig); h_orig = min(h_orig, height - y_orig)

                # --- SWIN ---
                try:
                    face_img = frame[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig]
                    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    inp = self.transform(pil).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        prob = torch.sigmoid(self.model(inp)).flatten().item()
                except: prob = 0.0

                # --- GEOMETRY ---
                ear_prob, mar_prob = 0.0, 0.0
                raw_ear, raw_mar = 0.0, 0.0
                landmarks_to_draw = []

                if self.dlib_ready:
                    shape = self.predictor(gray, rect)
                    shape_np = np.zeros((68, 2), dtype="int")
                    for i in range(0, 68):
                        shape_np[i] = (shape.part(i).x, shape.part(i).y)

                    leftEye = shape_np[36:42]
                    rightEye = shape_np[42:48]
                    mouth = shape_np[48:68]

                    leftEAR = self.eye_aspect_ratio(leftEye)
                    rightEAR = self.eye_aspect_ratio(rightEye)
                    raw_ear = (leftEAR + rightEAR) / 2.0
                    raw_mar = self.mouth_aspect_ratio(mouth)

                    # Logic: Convert to Fatigue Probability
                    # EAR: Normal is ~0.35. Drowsy is < 0.25.
                    if raw_ear < self.EYE_AR_THRESH: ear_prob = 1.0 
                    else: ear_prob = max(0, (0.35 - raw_ear) / 0.1)

                    # MAR: Normal is < 0.3. Yawn is > 0.6.
                    if raw_mar > self.MAR_THRESH: mar_prob = 1.0
                    else: mar_prob = max(0, (raw_mar - 0.3) / 0.3)
                    
                    landmarks_to_draw = [leftEye, rightEye, mouth]

                # --- FUSION FORMULA ---
                consensus = (prob * 0.50) + (ear_prob * 0.30) + (mar_prob * 0.20)
                
                self.prediction_buffer.append(consensus)
                avg = sum(self.prediction_buffer) / len(self.prediction_buffer)
                
                status = "FATIGUE" if avg > 0.5 else "Safe"
                color = (0, 0, 255) if avg > 0.5 else (0, 255, 0)

                current_results.append({
                    "box": (x_orig, y_orig, w_orig, h_orig),
                    "status": status, "score": avg, "color": color,
                    "swin": prob, 
                    "ear_val": raw_ear, "ear_prob": ear_prob,
                    "mar_val": raw_mar, "mar_prob": mar_prob,
                    "landmarks": landmarks_to_draw, "ratio": ratio
                })

            self.last_results = current_results

        # --- DRAWING ---
        for res in self.last_results:
            self.draw_dashboard(frame, res)
            
            if res["box"] is not None:
                x, y, w, h = res["box"]
                color = res["color"]
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                if self.dlib_ready:
                    ratio = res["ratio"]
                    for feat in res["landmarks"]:
                        for (lx, ly) in feat:
                            rx, ry = int(lx / ratio), int(ly / ratio)
                            cv2.circle(frame, (rx, ry), 2, (0, 255, 255), -1)

        # Resize for browser
        h, w = frame.shape[:2]
        if w > 800:
            scale = 800 / w
            frame = cv2.resize(frame, (800, int(h * scale)))

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()