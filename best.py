import torch
import os
import cv2
import time
from datetime import datetime
import supervision as sv


# ====== Reusable Functions (bisa di-import dari app.py) ======

def load_model(conf=0.25, iou=0.25):
    """Load YOLOv5 custom model (best3.pt)."""
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    yolov5_dir = os.path.join(curr_dir, 'YOLOv5', 'yolov5')
    best_pt = os.path.join(curr_dir, 'final.pt')
    model = torch.hub.load(yolov5_dir, 'custom', path=best_pt, source='local')
    model.conf = conf
    model.iou = iou
    model.classes = [0]  # Hanya detect 'person' (class 0 di COCO)
    return model


def create_tracker():
    """Buat ByteTrack tracker dan annotators."""
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    return tracker, box_annotator, label_annotator


def detect_and_track(model, frame, tracker, tracker_state, expire_timeout=300):
    """Jalankan deteksi + tracking pada satu frame.
    
    tracker_state: dict per tracker_id berisi:
        - first_seen: float (time.time saat pertama kali muncul)
        - last_seen: float (time.time saat terakhir terdeteksi)
        - clock_start: str (jam mulai, format HH:MM:SS)
        - clock_end: str (jam terakhir terdeteksi)
    
    expire_timeout: detik sebelum stopwatch dianggap expired (default 300 = 5 menit)
    
    Returns: (detections, labels, tracker_details, person_count)
    """
    results = model(frame)
    dets = results.xyxy[0].cpu().numpy()

    if len(dets) > 0:
        detections = sv.Detections(
            xyxy=dets[:, :4],
            confidence=dets[:, 4],
            class_id=dets[:, 5].astype(int)
        )
    else:
        detections = sv.Detections.empty()

    detections = tracker.update_with_detections(detections)

    current_time = time.time()
    now_str = datetime.now().strftime("%H:%M:%S")

    # Update tracker_state untuk yang terdeteksi sekarang
    current_ids = set()
    if detections.tracker_id is not None:
        for tid in detections.tracker_id:
            current_ids.add(int(tid))
            if tid not in tracker_state:
                tracker_state[tid] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'clock_start': now_str,
                    'clock_end': now_str,
                }
            else:
                tracker_state[tid]['last_seen'] = current_time
                tracker_state[tid]['clock_end'] = now_str

    # Labels untuk on-frame annotation (hanya yang terdeteksi sekarang)
    labels = []
    if detections.tracker_id is not None:
        for tracker_id, conf in zip(detections.tracker_id, detections.confidence):
            state = tracker_state[tracker_id]
            duration = current_time - state['first_seen']
            mins, secs = divmod(int(duration), 60)
            hrs, mins = divmod(mins, 60)
            labels.append(f"#{tracker_id} person {conf:.2f} | {hrs:02d}:{mins:02d}:{secs:02d}")

    # tracker_details untuk sidebar — SEMUA tracker yang belum expired
    tracker_details = []
    for tid, state in sorted(tracker_state.items()):
        time_since_last = current_time - state['last_seen']
        if time_since_last > expire_timeout:
            continue  # expired, skip

        duration = current_time - state['first_seen']
        mins, secs = divmod(int(duration), 60)
        hrs, mins = divmod(mins, 60)

        clock_range = f"{state['clock_start']} — {state['clock_end']}"

        if int(tid) in current_ids:
            status = "🟢"  # sedang terdeteksi
        else:
            status = "🟡"  # tidak terdeteksi tapi stopwatch masih jalan

        tracker_details.append(
            f"{status} **#{tid}** — {hrs:02d}:{mins:02d}:{secs:02d} — 🕐 {clock_range}"
        )

    person_count = len(detections)
    return detections, labels, tracker_details, person_count


def annotate_frame(frame, detections, labels, box_annotator, label_annotator,
                   person_count):
    """Annotate frame dengan bounding box, label, dan teks info."""
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    cv2.putText(frame, f"Person Count: {person_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame


# ====== Main (standalone mode via cv2) ======

def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(curr_dir, 'sample2.mp4')

    model = load_model(conf=0.25, iou=0.25)
    tracker, box_annotator, label_annotator = create_tracker()
    tracker_state = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka video!")
        return

    print("Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        detections, labels, _, person_count = detect_and_track(model, frame, tracker, tracker_state)

        frame = annotate_frame(frame, detections, labels, box_annotator, label_annotator,
                               person_count)

        cv2.imshow("YOLOv5 + ByteTrack", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
