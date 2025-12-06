import cv2
import dlib
import torch
import numpy as np
import os
from imutils import face_utils
from scipy.spatial import distance as dist
from torchvision import transforms
from PIL import Image
from collections import deque
import torchvision.models as models
import torch.nn as nn

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
MODEL_PATH = os.path.join(BASE_DIR, "swin_best.pth")

# --- 3D MODEL POINTS ---
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype=np.float64)

# --- HELPER FUNCTIONS ---
def get_head_pose(shape, img_h, img_w):
    image_points = np.array([
        shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]
    ], dtype="double")
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, jac = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1], angles[2], (int(image_points[0][0]), int(image_points[0][1]))

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])
    D = dist.euclidean(mouth[0], mouth[4])
    if D == 0: return 0
    return (A + B + C) / (3.0 * D)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    if C == 0: return 0
    return (A + B) / (2.0 * C)

class VideoCamera(object):
    def __init__(self, source=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.swin_v2_s(weights=None)
        self.model.head = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(in_features=768, out_features=1, bias=True))
        
        if os.path.exists(MODEL_PATH):
            try:
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device), strict=False)
                print(f"[INFO] Model loaded from {MODEL_PATH}")
            except Exception as e: print(f"[ERROR] Failed to load model: {e}")
        else:
            parent_path = os.path.join(BASE_DIR, "../saved_models/SwinTrans/swin_best.pth")
            if os.path.exists(parent_path):
                self.model.load_state_dict(torch.load(parent_path, map_location=self.device), strict=False)
                print(f"[INFO] Found model in parent directory")
            else: print(f"[WARNING] Model not found. Using random weights!")
            
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.detector = dlib.get_frontal_face_detector()
        try: self.predictor = dlib.shape_predictor(PREDICTOR_PATH)
        except: 
            parent_pred = os.path.join(BASE_DIR, "../shape_predictor_68_face_landmarks.dat")
            if os.path.exists(parent_pred): self.predictor = dlib.shape_predictor(parent_pred)
            else: print("CRITICAL: Shape predictor not found!")

        self.is_calibrating, self.calibration_frames, self.calibration_limit = True, 0, 45
        self.ear_readings, self.mar_readings, self.pitch_readings, self.roll_readings = [], [], [], []
        
        # Resting EAR baseline
        self.RESTING_EAR = 0.30 
        self.EYE_AR_THRESH, self.MOUTH_AR_THRESH, self.RESTING_PITCH, self.RESTING_ROLL, self.HEAD_THRESH = 0.25, 0.60, 0.0, 0.0, 15.0
        
        self.prediction_buffer = deque(maxlen=30)
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.mStart, self.mEnd = (60, 68)
        self.video = cv2.VideoCapture(source)

    def __del__(self): self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success: return None
        frame = cv2.resize(frame, (800, 600))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_h, img_w = frame.shape[:2]
        rects = self.detector(gray, 0)
        status, color = "Scanning...", (0, 255, 0)
        
        if len(rects) > 0:
            rect = rects[0]
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            leftEAR = eye_aspect_ratio(shape[self.lStart:self.lEnd])
            rightEAR = eye_aspect_ratio(shape[self.rStart:self.rEnd])
            ear = (leftEAR + rightEAR) / 2.0
            mar = mouth_aspect_ratio(shape[self.mStart:self.mEnd])
            pitch, yaw, roll, nose_point = get_head_pose(shape, img_h, img_w)

            if self.is_calibrating:
                self.ear_readings.append(ear); self.mar_readings.append(mar); self.pitch_readings.append(pitch); self.roll_readings.append(roll)
                self.calibration_frames += 1
                cv2.putText(frame, f"CALIBRATING... {self.calibration_frames}/{self.calibration_limit}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 255), 2)
                if self.calibration_frames >= self.calibration_limit:
                    self.is_calibrating = False
                    
                    self.RESTING_EAR = sum(self.ear_readings)/len(self.ear_readings)
                    
                    self.EYE_AR_THRESH = self.RESTING_EAR * 0.80
                    self.MOUTH_AR_THRESH = (sum(self.mar_readings)/len(self.mar_readings)) + 0.15
                    self.RESTING_PITCH = sum(self.pitch_readings)/len(self.pitch_readings)
                    self.RESTING_ROLL = sum(self.roll_readings)/len(self.roll_readings)
                    print(f"[CALIBRATION DONE] Resting EAR: {self.RESTING_EAR:.3f}, Threshold: {self.EYE_AR_THRESH:.3f}")
            else:
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                pad = 10
                face_img = frame[max(0, y-pad):min(img_h, y+h+pad), max(0, x-pad):min(img_w, x+w+pad)]
                if face_img.size > 0:
                    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        self.prediction_buffer.append(torch.sigmoid(output).item())
                
                avg_prob = sum(self.prediction_buffer)/len(self.prediction_buffer) if self.prediction_buffer else 0.5
                pitch_diff, roll_diff = abs(pitch - self.RESTING_PITCH), abs(roll - self.RESTING_ROLL)
                head_score = 1.0 if (pitch_diff > self.HEAD_THRESH or roll_diff > self.HEAD_THRESH) else 0.0

                # --- FIX: Calculate Droop Score BEFORE decision logic ---
                # So it is available for display even if Eyes Closed trigger fires
                droop_denom = (self.RESTING_EAR - self.EYE_AR_THRESH)
                # Avoid division by zero if resting ear is somehow same as thresh
                if droop_denom == 0: droop_denom = 0.001
                
                droop_score = (self.RESTING_EAR - ear) / droop_denom
                droop_score = max(0.0, min(droop_score, 1.0))

                if ear < self.EYE_AR_THRESH:
                    score, reason = 1.0, "EYES CLOSED"
                    droop_score = 1.0 # Force max droop for display
                elif head_score > 0.5:
                    score, reason = 1.0, "HEAD NOD/TILT"
                else:
                    # Formula: AI(60%) + Droop(20%) + Yawn(20%)
                    mar_score = 1.0 if mar > self.MOUTH_AR_THRESH else 0.0
                    score = (avg_prob * 0.6) + (droop_score * 0.2) + (mar_score * 0.2)
                    reason = "FATIGUE" if score > 0.5 else "Active"

                if score > 0.5: status, color = f"WARNING: {reason}", (0, 0, 255)
                else: status, color = "Active", (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"AI: {avg_prob:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                # Now this variable is guaranteed to exist
                cv2.putText(frame, f"Droop: {droop_score:.2f} (EAR: {ear:.2f})", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(frame, f"Score: {score:.2f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                p2 = (int(nose_point[0] + yaw * 2), int(nose_point[1] - pitch * 2))
                cv2.line(frame, nose_point, p2, (0, 255, 255), 2)

            for (x, y) in shape: cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()