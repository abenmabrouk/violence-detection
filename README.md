# Violence Detection in Videos

This project detects violent actions in videos using a **3D Convolutional Neural Network (3D CNN)** based on **ResNet-18 (r3d_18)**.
The model is trained on the **Violent Flows dataset** and can perform **real-time violence detection** on video files or webcam streams.

## Project Structure

- `data/` : Contains the training and validation videos (not included in repo). Please refer to the following link to access the dataset: https://www.kaggle.com/api/v1/datasets/download/kabeleswarpe/violent-flows. 
- `scripts/convert_videos.py` : Converts `.avi` videos to `.mp4`.  
- `src/` : Core scripts:
  - `train.py` : Training script
  - `detect_video.py` : Detect violence in a video or webcam feed
  - `utils.py` : Helper functions (read video as tensor, etc.)

## Installation

1. Clone the repo:
```bash
git clone https://github.com/abenmabrouk/violence-detection.git
cd violence-detection
```
2. Create a virtual environment :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
## Usage
1. Training
``` bash
python scr/train.py
```
2. Detect violence in a video
```bash
python src/detect_video.py path/to/video.mp4
```
3. Detect violence from webcam
``` bash
python src/detect_video.py
```

**Notes** 

* Model input: 16 frames per clip, resized to 112x112.

* Output: Violence or No Violence label.

* Pretrained model **r3d_18** is used.