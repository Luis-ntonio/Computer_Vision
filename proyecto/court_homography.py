import cv2, numpy as np
from ultralytics import YOLO

class CourtHomography:
    def __init__(self, model_path, real_pts):
        self.model = YOLO(model_path)
        self.H = None
        self.real_pts = np.array(real_pts, dtype=np.float32)

    def compute(self, img):
        res = self.model.predict(source=img, task='pose', imgsz=640, save=False)[0]
        pts = res.keypoints.xy  # pixel coords N×2
        if pts.shape[0] >= self.real_pts.shape[0]:
            self.H, _ = cv2.findHomography(pts[:len(self.real_pts)], self.real_pts)
        return self.H
