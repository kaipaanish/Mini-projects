import os
import cv2
import numpy as np

def _vertical_edge_density(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    return np.mean(np.abs(gx) > 30)

def _heuristic_detect(image_path: str, cfg: dict):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # emphasize vertical edges
    kx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edges = cv2.convertScaleAbs(np.abs(kx))
    _, bw = cv2.threshold(edges, 0, 255, cv2.THRESH_OTSU)

    # Close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_ar = float(cfg.get("min_aspect_ratio", 1.6))
    min_area = float(cfg.get("min_area_frac", 0.01)) * h * w
    max_area = float(cfg.get("max_area_frac", 0.25)) * h * w

    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < min_area or area > max_area:
            continue
        ar = ch / max(cw, 1)
        if ar < min_ar:
            continue
        candidates.append((x, y, cw, ch))

    detections = []
    canvas = img.copy()
    for (x, y, cw, ch) in candidates:
        roi = gray[y:y+ch, x:x+cw]
        mean_gray = float(np.mean(roi))
        edge_density = _vertical_edge_density(roi)
        # Simple rule:
        # - very dark mean -> likely open
        # - otherwise if edge-dense -> likely closed
        if mean_gray < float(cfg.get("open_threshold", 110)):
            label = "door_open"
            color = (0, 255, 0)
        elif edge_density > float(cfg.get("edge_density_closed", 0.05)):
            label = "door_closed"
            color = (0, 128, 255)
        else:
            label = "door(?)"
            color = (255, 255, 0)
        detections.append((label, (x, y, cw, ch)))
        cv2.rectangle(canvas, (x, y), (x+cw, y+ch), color, 2)
        cv2.putText(canvas, label, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return canvas, [d[0] for d in detections]

def detect_doors(image_path: str, out_path: str, yolo_model_path: str, heuristic_cfg: dict):
    detections = []
    img = None

    if yolo_model_path and os.path.exists(yolo_model_path):
        try:
            from ultralytics import YOLO
            model = YOLO(yolo_model_path)
            results = model(image_path, verbose=False)
            res = results[0]
            img = res.plot()  # annotated image
            for box, cls_idx in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)):
                cls_name = model.names.get(cls_idx, str(cls_idx))
                detections.append(cls_name)
        except Exception as e:
            print(f"[warn] YOLO failed ({e}), falling back to heuristic.")
            img, detections = _heuristic_detect(image_path, heuristic_cfg)
    else:
        img, detections = _heuristic_detect(image_path, heuristic_cfg)

    # Save annotated
    if img is None:
        img = cv2.imread(image_path)
    cv2.imwrite(out_path, img)
    return detections
