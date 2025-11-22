import torch
import torchvision
import torch.nn as nn

from FatigueDetectionDemo import SwinT, CHECKPOINT_PATH
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from torchvision import transforms
import pandas as pd

# 1. Instantiate the Swin backbone
swin_b_32 = torchvision.models.swin_v2_s(weights=True)
swin_b_32.head = nn.Sequential(
    nn.Linear(in_features=768, out_features=1, bias=True),
    nn.Sigmoid(),
)

# 2. Create the LightningModule
model = SwinT(swin=swin_b_32)

# 3. Load the saved weights
model.model.load_state_dict(torch.load(f"{CHECKPOINT_PATH}/swin_best.pth", map_location="cpu"))
model.eval()


# 4. Predict on user-specified images

# --- Interactive GUI with tkinter and pandas ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        prob = output.flatten().item()
        pred = int(prob > 0.5)
    return pred, prob

def show_result(img_path, pred, prob):
    # Show image and result in a new window
    result_window = tk.Toplevel()
    result_window.title("Prediction Result")
    # Show image
    img = Image.open(img_path).resize((224, 224))
    tk_img = ImageTk.PhotoImage(img)
    img_label = tk.Label(result_window, image=tk_img)
    img_label.image = tk_img
    img_label.pack()
    # Show pandas DataFrame
    df = pd.DataFrame({
        'Image': [img_path],
        'Predicted Class': [pred],
        'Probability': [prob]
    })
    text = tk.Text(result_window, height=5, width=80)
    text.insert(tk.END, df.to_string(index=False))
    text.config(state=tk.DISABLED)
    text.pack()

def select_and_predict():
    img_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    if img_path:
        try:
            pred, prob = predict_image(img_path)
            show_result(img_path, pred, prob)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{e}")

if __name__ == "__main__":
    model.eval()
    root = tk.Tk()
    root.title("Fatigue Detection Demo - Image Predictor")
    root.geometry("600x400")
    btn = tk.Button(root, text="Select Image and Predict", command=select_and_predict, font=("Arial", 14))
    btn.pack(expand=True)
    root.mainloop()
