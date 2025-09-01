import os
import cv2
import numpy as np

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def list_videos(folder: str):
    exts = (".mp4", ".mov", ".avi", ".mkv")
    for name in os.listdir(folder):
        if name.lower().endswith(exts):
            yield os.path.join(folder, name)

def moving_average(x, w):
    if w <= 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[w:] - cumsum[:-w]) / float(w)

def time_from_frames(idx: int, fps: float) -> float:
    return max(0.0, idx / float(fps))

def write_debug_plot_png(xs, ys, out_path: str, title: str):
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(xs, ys)
        plt.title(title)
        plt.xlabel("frame")
        plt.ylabel("foreground ratio")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[warn] failed to write debug plot ({e})")
