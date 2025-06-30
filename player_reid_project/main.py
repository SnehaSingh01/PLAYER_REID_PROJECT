import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Paths
VIDEO_PATH = 'data/15sec_input_720p.mp4'
MODEL_PATH = 'models/best.pt'
OUTPUT_PATH = 'outputs/reid_output.mp4'

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

# Initialize DeepSort Tracker
tracker = DeepSort(
    max_age=60,              # allow more time for re-identification
    n_init=3,
    max_cosine_distance=0.3
)

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = []
    results = model(frame)[0]

    # Confidence threshold to reduce false positives
    CONF_THRESH = 0.35

    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if conf < CONF_THRESH:
            continue

        # Filter likely non-player classes using height-width ratio
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1

        aspect_ratio = h / (w + 1e-6)
        if aspect_ratio < 1.3:  # Skip low-height boxes (likely footballs or other noise)
            continue

        detections.append(([x1, y1, w, h], conf.item(), 'player'))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracked boxes
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Player {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Player Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
