import cv2
import numpy as np
import itertools

# ——————————————
# Clase KalmanTracker (igual que antes)
# ——————————————
class KalmanTracker:
    _id_iter = itertools.count()

    def __init__(self, initial_bbox):
        x, y, w, h = initial_bbox
        cx = x + w / 2.
        cy = y + h / 2.

        self.id = next(KalmanTracker._id_iter)
        self.kf = cv2.KalmanFilter(4, 2)
        # Matriz de transición (Δt = 1)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        # Medición: solo posición (cx, cy)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        # Covarianzas de ruido (tuning opcional)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        # Estado inicial: [cx, cy, vx=0, vy=0]
        self.kf.statePost = np.array([[cx], [cy], [0.], [0.]], dtype=np.float32)

        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0
        self.bbox = initial_bbox

    def predict(self):
        pred = self.kf.predict()
        cx, cy = float(pred[0]), float(pred[1])
        self.age += 1
        self.time_since_update += 1
        return cx, cy

    def update(self, bbox):
        x, y, w, h = bbox
        cx = x + w / 2.
        cy = y + h / 2.
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kf.correct(measurement)
        self.bbox = bbox
        self.time_since_update = 0
        self.hit_streak += 1

    def get_state(self):
        state = self.kf.statePost
        cx, cy = float(state[0]), float(state[1])
        x = int(cx - self.bbox[2] / 2.)
        y = int(cy - self.bbox[3] / 2.)
        return (x, y, self.bbox[2], self.bbox[3])