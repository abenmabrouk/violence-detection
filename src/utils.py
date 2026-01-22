import cv2
import torch
import numpy as np

def read_clip_as_tensor(video_path, frames_per_clip=16, size=(112, 112)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < frames_per_clip:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, size)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from: {video_path}")
    while len(frames) < frames_per_clip:
        frames.append(frames[-1])
    clip = torch.tensor(np.stack(frames)).permute(3, 0, 1, 2) / 255.0  # (C, T, H, W)
    return clip.unsqueeze(0).float()  # (1, C, T, H, W)
