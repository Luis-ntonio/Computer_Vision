import cv2, pandas as pd
from ultralytics import YOLO
import numpy as np

class PeopleBallTracker:
    def __init__(self, model_path, H=None):
        self.model = YOLO(model_path)
        self.H = H
        self.records = []

    def process_frame(self, img, frame_idx):
        res = self.model.track(source=img, task='detect', imgsz=640, persist=True, stream=True).__next__()
        annotated = res.plot()
        #print(f"Frame {frame_idx}: boxes has attributes {dir(res.boxes)}")
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes.xyxy is not None else None
        ids = res.boxes.id.cpu().numpy() if res.boxes.id is not None else None
        if boxes is None or ids is None or len(boxes) == 0 or len(ids) == 0:
            return annotated
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        if self.H is not None:
            centers_real = cv2.perspectiveTransform(centers.reshape(-1,1,2), self.H).reshape(-1,2)
        else:
            centers_real = centers
        for cid, cen, cen_r in zip(ids, centers, centers_real):
            self.records.append({
                'frame': frame_idx, 'id': int(cid),
                'x': cen[0], 'y': cen[1],
                'x_real': cen_r[0], 'y_real': cen_r[1]
            })
        return annotated

    def save_csv(self, path):
        pd.DataFrame(self.records).to_csv(path, index=False)
