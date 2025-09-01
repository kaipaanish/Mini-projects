import os
import sys
import argparse
import yaml
from datetime import datetime
from glob import glob
from tqdm import tqdm

from .utils import ensure_dir, stem
from .coach_splitter import CoachSplitter
from .frame_extractor import extract_frames_from_video
from .door_detector import detect_doors
from .report_builder import build_html_report, build_pdf_report

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def stage_split(cfg):
    train_number = int(cfg["train_number"])
    input_video = cfg["input_video"]
    splitter_cfg = cfg.get("splitter", {})
    out_root = os.path.join("outputs", "Processed_Video", str(train_number))
    ensure_dir(out_root)

    sp = CoachSplitter(cfg)
    coach_dirs, count, boundaries = sp.split(input_video, out_root, train_number, splitter_cfg)
    # write summary
    with open(os.path.join(out_root, "split_summary.txt"), "w") as f:
        f.write(f"coach_count={count}\n")
        f.write(f"boundaries_frames={boundaries}\n")
    print(f"[split] coaches found: {count}")
    return coach_dirs

def stage_frames(cfg, coach_dirs=None):
    train_number = int(cfg["train_number"])
    frames_per_coach = int(cfg.get("frames_per_coach", 5))
    out_root = os.path.join("outputs", "Processed_Video", str(train_number))

    if coach_dirs is None:
        # discover coach dirs
        coach_dirs = sorted([d for d in glob(os.path.join(out_root, f"{train_number}_*")) if os.path.isdir(d)])

    for coach_dir in tqdm(coach_dirs, desc="[frames] coaches"):
        coach_idx = int(os.path.basename(coach_dir).split("_")[-1])
        video_path = os.path.join(coach_dir, f"{train_number}_{coach_idx}.mp4")
        frames_dir = os.path.join(coach_dir, "frames")
        ensure_dir(frames_dir)
        saved = extract_frames_from_video(video_path, frames_dir, train_number, coach_idx, frames_per_coach)
        print(f"[frames] coach {coach_idx}: saved {len(saved)} frames")

    return coach_dirs

def stage_detect(cfg, coach_dirs=None):
    train_number = int(cfg["train_number"])
    out_root = os.path.join("outputs", "Processed_Video", str(train_number))
    yolo_model_path = cfg.get("yolo_model_path", "")
    heuristic_cfg = cfg.get("heuristic", {})

    if coach_dirs is None:
        coach_dirs = sorted([d for d in glob(os.path.join(out_root, f"{train_number}_*")) if os.path.isdir(d)])

    for coach_dir in tqdm(coach_dirs, desc="[detect] coaches"):
        coach_idx = int(os.path.basename(coach_dir).split("_")[-1])
        frames_dir = os.path.join(coach_dir, "frames")
        anno_dir = os.path.join(coach_dir, "frames_annotated")
        ensure_dir(anno_dir)

        for img_name in sorted(os.listdir(frames_dir)):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            img_path = os.path.join(frames_dir, img_name)
            out_path = os.path.join(anno_dir, img_name)
            _ = detect_doors(img_path, out_path, yolo_model_path, heuristic_cfg)

    return coach_dirs

def stage_report(cfg, coach_dirs=None):
    train_number = int(cfg["train_number"])
    input_video = cfg["input_video"]
    out_root = os.path.join("outputs", "Processed_Video", str(train_number))
    report_dir = os.path.join("reports", "side_view", str(train_number))
    ensure_dir(report_dir)

    if coach_dirs is None:
        coach_dirs = sorted([d for d in glob(os.path.join(out_root, f"{train_number}_*")) if os.path.isdir(d)])

    coaches_ctx = []
    for coach_dir in coach_dirs:
        coach_idx = int(os.path.basename(coach_dir).split("_")[-1])
        video_name = f"{train_number}_{coach_idx}.mp4"
        anno_dir = os.path.join(coach_dir, "frames_annotated")
        frames = []
        for img_name in sorted(os.listdir(anno_dir)):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            path = os.path.join(anno_dir, img_name)
            # Try to read detection label(s) from filename pattern (optional)
            detections = []
            frames.append({
                "name": img_name,
                "anno_path": path,
                "detections": detections
            })
        coaches_ctx.append({
            "index": coach_idx,
            "video_name": video_name,
            "frames": frames
        })

    context = {
        "train_number": train_number,
        "coach_count": len(coach_dirs),
        "input_video": input_video,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "coaches": sorted(coaches_ctx, key=lambda x: x["index"]) 
    }

    out_html = os.path.join(report_dir, "report.html")
    out_pdf = os.path.join(report_dir, "report.pdf")

    build_html_report("templates", out_html, context)
    build_pdf_report(out_pdf, context)
    print(f"[report] HTML: {out_html}")
    print(f"[report] PDF : {out_pdf}")
    return out_html, out_pdf

def run_all(cfg):
    coach_dirs = stage_split(cfg)
    coach_dirs = stage_frames(cfg, coach_dirs)
    coach_dirs = stage_detect(cfg, coach_dirs)
    _ = stage_report(cfg, coach_dirs)

def main():
    ap = argparse.ArgumentParser(description="Train side-view processing pipeline")
    ap.add_argument("stage", choices=["split", "frames", "detect", "report", "all"])
    ap.add_argument("--config", "-c", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..")  # go to project root

    if args.stage == "split":
        stage_split(cfg)
    elif args.stage == "frames":
        stage_frames(cfg)
    elif args.stage == "detect":
        stage_detect(cfg)
    elif args.stage == "report":
        stage_report(cfg)
    elif args.stage == "all":
        run_all(cfg)

if __name__ == "__main__":
    main()
