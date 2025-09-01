import os
import math
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from .utils import ensure_dir, stem, moving_average, time_from_frames, write_debug_plot_png

class CoachSplitter:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _compute_foreground_signal(self, cap: cv2.VideoCapture, sample_step: int, roi_top: float, roi_bottom: float,
                                   history: int, var_threshold: int):
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        y1 = int(height * roi_top)
        y2 = int(height * roi_bottom)
        mog2 = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=False)

        vals = []
        frame_ids = []

        for i in range(0, length, sample_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                break
            roi = frame[y1:y2, :]
            fg = mog2.apply(roi)
            # foreground ratio inside ROI
            fg_ratio = (fg > 0).mean()
            vals.append(fg_ratio)
            frame_ids.append(i)

        return np.array(frame_ids), np.array(vals), fps

    def _find_gaps(self, xs, ys, smooth_window: int, min_gap_len: int, min_segment_seconds: float, fps: float):
        # smooth
        y_s = ys.copy()
        if smooth_window > 1 and smooth_window < len(ys):
            y_s = moving_average(ys, smooth_window)
            xs = xs[:len(y_s)]

        # threshold at mid of (min, median)
        lo = float(np.min(y_s))
        md = float(np.median(y_s))
        thr = (lo + md) / 2.0

        below = y_s < thr
        gaps = []
        start = None
        for i, b in enumerate(below):
            if b and start is None:
                start = i
            elif (not b) and start is not None:
                end = i - 1
                if end - start + 1 >= min_gap_len:
                    gaps.append((start, end))
                start = None
        if start is not None:
            end = len(below) - 1
            if end - start + 1 >= min_gap_len:
                gaps.append((start, end))

        # convert to frame indices (use center of valley as boundary)
        boundaries = []
        for a, b in gaps:
            c = xs[a + (b - a) // 2]
            boundaries.append(int(c))

        # ensure min segment duration
        boundaries2 = []
        last_boundary_frame = 0
        min_len_frames = int(min_segment_seconds * fps)
        for b in boundaries:
            if b - last_boundary_frame >= min_len_frames:
                boundaries2.append(b)
                last_boundary_frame = b
        return boundaries2, y_s, xs

    def split(self, video_path: str, out_dir: str, train_number: int, params: dict):
        ensure_dir(out_dir)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        sample_step = int(params.get("sample_step", 2))
        roi_top = float(params.get("roi_top", 0.35))
        roi_bottom = float(params.get("roi_bottom", 0.65))
        history = int(params.get("mog2_history", 400))
        var_threshold = int(params.get("mog2_var_threshold", 16))
        smooth_window = int(params.get("smooth_window", 21))
        min_gap_len = int(params.get("min_gap_len", 10))
        min_segment_seconds = float(params.get("min_segment_seconds", 2.5))
        debug_plots = bool(params.get("debug_plots", False))

        xs, ys, fps = self._compute_foreground_signal(cap, sample_step, roi_top, roi_bottom, history, var_threshold)
        boundaries, y_s, xs_s = self._find_gaps(xs, ys, smooth_window, min_gap_len, min_segment_seconds, fps)

        # write debug plot
        if debug_plots:
            plot_path = os.path.join(out_dir, "foreground_signal.png")
            write_debug_plot_png(xs_s, y_s, plot_path, "Foreground signal (valleys ~= inter-coach gaps)")

        # Real splitting using MoviePy
        cap.release()
        clip = VideoFileClip(video_path)
        total_frames = int(clip.fps * clip.duration)

        frames = [0] + boundaries + [total_frames - 1]
        segments = []
        for i in range(len(frames) - 1):
            a = frames[i]
            b = frames[i+1]
            # guard against zero length
            if b <= a:
                continue
            t1 = max(0.0, a / clip.fps)
            t2 = max(t1 + 0.10, b / clip.fps)  # ensure small positive duration
            segments.append((t1, t2))

        coach_dirs = []
        for idx, (t1, t2) in enumerate(segments, 1):
            coach_dir = os.path.join(out_dir, f"{train_number}_{idx}")
            ensure_dir(coach_dir)
            out_video = os.path.join(coach_dir, f"{train_number}_{idx}.mp4")
            sub = clip.subclip(t1, t2)
            sub.write_videofile(out_video, codec="libx264", audio=False, verbose=False, logger=None)
            coach_dirs.append(coach_dir)

        clip.close()

        return coach_dirs, len(segments), boundaries
