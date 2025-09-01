import os
import cv2
from .utils import ensure_dir

def extract_frames_from_video(video_path: str, out_dir: str, train_number: int, coach_index: int, frames_per_coach: int):
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # positions as percents across the clip
    positions = [int(frame_count * p) for p in [i/(frames_per_coach+1) for i in range(1, frames_per_coach+1)]]

    saved = []
    for j, pos in enumerate(positions, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            continue
        out_name = f"{train_number}_{coach_index}_{j}.jpg"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, frame)
        saved.append(out_path)
    cap.release()
    return saved
