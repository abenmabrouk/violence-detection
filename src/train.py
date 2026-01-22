import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models.video import r3d_18
from torch.utils.data import DataLoader
import os
import cv2
import glob
from tqdm import tqdm

# --- Dataset ---
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, frames_per_clip=16):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.samples = []
        classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        for cls in classes:
            videos = glob.glob(os.path.join(root_dir, cls, "*.mp4"))
            for vid in videos:
                self.samples.append((vid, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def read_video_frames(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < self.frames_per_clip:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
        cap.release()
        while len(frames) < self.frames_per_clip:
            frames.append(frames[-1])  # repeat last frame
        return torch.tensor(frames).permute(3,0,1,2) / 255.0

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        clip = self.read_video_frames(video_path)
        if self.transform:
            clip = self.transform(clip)
        return clip.float(), label

# --- Transformations ---
transform = None

# --- DataLoader ---
train_dataset = VideoDataset("data/train", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# --- Model ---
model = r3d_18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # violence / nonviolence

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- Simple Training ---
epochs = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = model.to(device)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for clips, labels in tqdm(train_loader):
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f}")

# --- Saving ---
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/violence_model.pth")
print("Model saved!")
