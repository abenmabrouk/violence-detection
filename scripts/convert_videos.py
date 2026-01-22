import os
import subprocess
from pathlib import Path

# Path to your dataset folder
DATA_DIR = Path("data")

# video extensions to convert
SOURCE_EXT = ".avi"
TARGET_EXT = ".mp4"

def convert_video(input_path, output_path):
    """convert video mp4 with ffmpeg."""
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        str(output_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def process_dataset():
    """ Browse the entire dataset and convert all .avi to .mp4."""
    avi_count = 0
    for video_path in DATA_DIR.rglob(f"*{SOURCE_EXT}"):
        avi_count += 1
        output_path = video_path.with_suffix(TARGET_EXT)
        print(f"[INFO] Conversion: {video_path} -> {output_path}")
        convert_video(video_path, output_path)
        video_path.unlink()  # delete the original file .avi

    if avi_count == 0:
        print("[INFO] No video file .avi found, nothing to convert.")
    else:
        print(f"[INFO] Conversion completed : {avi_count} converted files.")

def count_videos():
    """  count the videos per class abd folder """
    print("\n Dataset Summary :")
    for folder in ["train", "val"]:
        for label in ["Violence", "NonViolence"]:
            path = DATA_DIR / folder / label
            if path.exists():
                files = list(path.glob(f"*{TARGET_EXT}"))
                print(f"{folder}/{label} : {len(files)} videos")
            else:
                print(f"[WARNING] Folder not found: {path}")

if __name__ == "__main__":
    process_dataset()
    count_videos()
