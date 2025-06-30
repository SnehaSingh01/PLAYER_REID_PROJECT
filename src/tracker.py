# src/tracker.py
import numpy as np

class SimpleTracker:
    def __init__(self, max_lost=30):
        self.next_id = 0
        self.players = {}  # player_id : [bbox, lost_frames]
        self.max_lost = max_lost
    
    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def update(self, detections):
        updated_players = {}
        
        for det in detections:
            x1, y1, x2, y2, conf = det
            matched_id = None
            best_iou = 0.0
            
            for player_id, (bbox, lost_frames) in self.players.items():
                iou_score = self.iou(bbox, [x1, y1, x2, y2])
                if iou_score > 0.3 and iou_score > best_iou:
                    matched_id = player_id
                    best_iou = iou_score
            
            if matched_id is not None:
                updated_players[matched_id] = ([x1, y1, x2, y2], 0)
            else:
                updated_players[self.next_id] = ([x1, y1, x2, y2], 0)
                self.next_id += 1
        
        # Handle lost players
        for player_id, (bbox, lost_frames) in self.players.items():
            if player_id not in updated_players:
                if lost_frames < self.max_lost:
                    updated_players[player_id] = (bbox, lost_frames + 1)
        
        self.players = updated_players
        return self.players
