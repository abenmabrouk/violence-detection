import cv2
import sys
import torch
from torchvision import transforms
from torchvision.models.video import r3d_18
import numpy as np
from pathlib import Path

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "models/violence_model.pth"
IMG_SIZE = (112, 112)  # same as train.py
FRAMES_PER_CLIP = 16
CLASSES = ["No Violence", "Violence"]
WINDOW_NAME = "Violence Detection"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)


# ==============================
# Model Downloading
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = r3d_18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ==============================
# PreProcessing
# ==============================
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, IMG_SIZE)
    frame = torch.tensor(frame).permute(2, 0, 1) / 255.0
    return frame

# ==============================
# Video source openning : video or webcam
# ==============================
if len(sys.argv) > 1:
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"[ERROR] Missing Video : {video_path}")
        sys.exit(1)
    cap = cv2.VideoCapture(str(video_path))
    print(f"[INFO] Reading video from {video_path}")
else:
    cap = cv2.VideoCapture(0)
    print("[INFO] No video source given → Webcam used.")

if not cap.isOpened():
    print("[ERROR] Impossible to open the video")
    sys.exit(1)

print("[INFO] Press 'q' to quit.")

# ==============================
# Processing the video
# ==============================
frames_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Video Source ended.")
        break

    display_frame = frame.copy()
    frames_buffer.append(preprocess_frame(frame))

    if len(frames_buffer) == FRAMES_PER_CLIP:
        clip_tensor = torch.stack(frames_buffer, dim=1).unsqueeze(0).to(device)
        frames_buffer.pop(0)

        with torch.no_grad():
            outputs = model(clip_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            label = CLASSES[pred_idx]
            confidence = probs[pred_idx].item()

        cv2.putText(
            display_frame,
            f"{label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if label == "Violence" else (0, 255, 0),
            2
        )

    cv2.imshow(WINDOW_NAME, display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break

cap.release()
cv2.destroyAllWindows()
