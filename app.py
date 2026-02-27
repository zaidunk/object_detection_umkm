import streamlit as st
import cv2
import numpy as np
import os
import tempfile

from best import load_model, create_tracker, detect_and_track, annotate_frame

st.set_page_config(page_title="YOLOv5 Person Tracker", layout="wide")
st.title("🎯 YOLOv5 + ByteTrack Person Tracking")
st.markdown("Upload video untuk mendeteksi dan melacak orang secara real-time.")

# --- Sidebar controls ---
st.sidebar.header("⚙️ Pengaturan")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)
skip_frames = st.sidebar.slider("Proses setiap N frame", 1, 5, 1, 1)
decay_factor = st.sidebar.slider("Decay Factor", 0.9, 1.0, 0.995, 0.001, format="%.3f")

# --- Model loading (cached) ---
@st.cache_resource
def get_model():
    return load_model()

model = get_model()
model.conf = confidence
model.iou = iou_threshold

# --- Video source ---
curr_dir = os.path.dirname(os.path.abspath(__file__))
default_video = os.path.join(curr_dir, 'data1.mp4')

source_option = st.radio(
    "Sumber Video",
    ["Upload Video", "Video Default (data1.mp4)"],
    horizontal=True
)

video_path = None
if source_option == "Upload Video":
    uploaded_file = st.file_uploader("Upload file video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        tfile.close()
else:
    if os.path.exists(default_video):
        video_path = default_video
    else:
        st.error("Video default tidak ditemukan: sample3.mp4")

# --- Processing ---
if video_path and st.button("▶️ Mulai Deteksi & Tracking", type="primary"):
    tracker, box_annotator, label_annotator = create_tracker()
    tracker_state = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Tidak bisa membuka video!")
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.info(f"📹 Video: {width}x{height} | {fps:.1f} FPS | {total_frames} frames")

        col1, col2 = st.columns([3, 1])
        with col1:
            frame_placeholder = st.empty()
        with col2:
            count_placeholder = st.empty()
            tracker_info_placeholder = st.empty()
            progress_placeholder = st.empty()

        progress_bar = st.progress(0)
        stop_button = st.button("⏹️ Stop")

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % skip_frames != 0:
                continue

            # Deteksi + tracking (dari best.py)
            detections, labels, tracker_details, person_count = detect_and_track(
                model, frame, tracker, tracker_state
            )

            # Annotate frame (dari best.py)
            frame = annotate_frame(frame, detections, labels, box_annotator, label_annotator,
                                   person_count)

            # Convert BGR -> RGB untuk Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Update display
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            count_placeholder.metric("👥 Person Count", person_count)
            tracker_info_placeholder.markdown(
                "**Tracked Persons:**\n" + "\n".join(tracker_details) if tracker_details else "*Tidak ada orang terdeteksi*"
            )

            progress = frame_idx / total_frames if total_frames > 0 else 0
            progress_bar.progress(min(progress, 1.0))
            progress_placeholder.caption(f"Frame {frame_idx}/{total_frames}")

            if stop_button:
                break

        cap.release()
        progress_bar.progress(1.0)
        st.success(f"✅ Selesai! Total unique persons tracked: **{len(tracker_state)}**")

elif not video_path:
    st.info("👆 Pilih sumber video untuk memulai.")
