# src/deepsort_tracker.py
from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30)

    def update_tracks(self, detections, frame=None):
        # Convert detections: [[x1, y1, x2, y2, conf]] → [[x1, y1, x2, y2, conf]]
        det_list = [[d[0], d[1], d[2], d[3], d[4]] for d in detections]
        tracks = self.tracker.update_tracks(det_list, frame=frame)
        return tracks
