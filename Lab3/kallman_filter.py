import cv2
import numpy as np
import itertools

# —————————————— 1) Clase KalmanTracker (igual que antes) ——————————————
class KalmanTracker:
    _id_iter = itertools.count()

    def __init__(self, initial_bbox):
        x, y, w, h = initial_bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        self.id = next(KalmanTracker._id_iter)
        self.bbox = initial_bbox  # (x, y, w, h) de la última detección
        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0

        # 1) Crear Kalman Filter con 4 estados (cx, cy, vx, vy) y 2 mediciones (cx, cy)
        self.kf = cv2.KalmanFilter(4, 2)

        # 2) Matriz de transición A (Δt = 1)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # 3) Matriz de medición H: solo posición (cx, cy)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # 4) Ruido de proceso Q (tuning opcional)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

        # 5) Ruido de medición R (tuning opcional)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        # 6) Covarianza inicial del error (P)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        # 7) Estado inicial: [cx, cy, vx=0, vy=0]
        self.kf.statePost = np.array([[cx], [cy], [0.], [0.]], dtype=np.float32)

    def predict(self):
        """
        Ejecuta la predicción del Kalman (paso a adelante Δt=1).
        Retorna el centro (cx, cy) predicho.
        """
        pred = self.kf.predict()
        cx, cy = float(pred[0]), float(pred[1])
        self.age += 1
        self.time_since_update += 1
        return cx, cy

    def update(self, bbox):
        """
        Corrige el filtro con la medición real.
        bbox = (x, y, w, h)
        """
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kf.correct(measurement)

        self.bbox = bbox
        self.time_since_update = 0
        self.hit_streak += 1

    def get_state(self):
        """
        Obtiene la caja actual estimada (x, y, w, h) a partir del estado del filtro.
        Utiliza el centro (cx, cy) del statePost y el w,h de la última detección.
        """
        state = self.kf.statePost  # 4×1 array: [cx, cy, vx, vy]^T
        cx, cy = float(state[0]), float(state[1])
        w, h = self.bbox[2], self.bbox[3]
        x = int(cx - w / 2.0)
        y = int(cy - h / 2.0)
        return (x, y, w, h)

    def predict_future(self, n):
        """
        Proyecta el estado del Kalman Filter n pasos hacia adelante (Δt = 1 en cada paso),
        sin modificar el estado interno real. Retorna lista de _n_ centros futuros [(cx1,cy1), (cx2,cy2), ...].
        """
        # Extraer el estado actual (statePost) como numpy array 4×1
        state = self.kf.statePost.copy()

        # Leer la matriz de transición A (ya es un numpy.ndarray 4×4)
        A = self.kf.transitionMatrix

        futuros = []
        s = state.copy()  # clon para no cambiar el filtro original

        for _ in range(n):
            # s_{k+1} = A * s_k
            s = A.dot(s)
            cx_futuro, cy_futuro = float(s[0]), float(s[1])
            futuros.append((cx_futuro, cy_futuro))

        return futuros