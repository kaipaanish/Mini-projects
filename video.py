import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
import time

st.set_page_config(page_title="Video Object Detection", layout="wide")

# Title
st.title("🎯 Real-Time Object Detection on Video")
st.markdown("Upload a `.mp4` file to visualize YOLOv8 frame-by-frame object detection.")

# Load YOLO model (do not load unless video is uploaded!)
uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file is not None:
    # Load model after video is uploaded
    model = YOLO("yolov8n.pt")

    # Save video to temp file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_file.read())
    video_path = temp_video.name

    st.video(video_path)  # Optional: show original video

    st.markdown("### Processed Frame-by-Frame Output")
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)

    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO
        results = model(frame)[0]
        annotated = results.plot()

        # Convert to RGB for Streamlit
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # Display frame
        frame_placeholder.image(annotated_rgb, channels="RGB")

        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

        # Optional delay to simulate live preview
        time.sleep(0.01)

    cap.release()
    st.success("✅ Done! Video processing completed.")
