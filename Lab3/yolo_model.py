from ultralytics import YOLO
from dotenv import load_dotenv
import os

load_dotenv()  # Carga variables de entorno desde .env (opcional)

MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")  # Nombre del modelo, por defecto yolov8n.pt


# 1) Carga el modelo (automáticamente descarga yolov8n.pt si no lo tienes)
def load_yolo_model():
    model = YOLO(MODEL)  # "n" = nano (ligero). También hay yolov8s.pt, yolov8m.pt, etc.
    return model

# 2) Función de detección que recibe un frame de OpenCV y devuelve la lista de bboxes
def detect_objects_ultralytics(frame, conf_threshold=0.25):
    """
    Ejecuta YOLOv8 sobre 'frame' y devuelve lista de detecciones:
    [(x, y, w, h, conf, class_id), ...]
    """
    model = load_yolo_model()
    results = model(frame)[0]  # Pipelines internos de Ultralytics
    detections = []
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        if float(conf) < conf_threshold:
            continue
        x1, y1, x2, y2 = box.cpu().numpy()
        x, y = int(x1), int(y1)
        w, h = int(x2 - x1), int(y2 - y1)
        detections.append((x, y, w, h, float(conf), int(cls)))
    return detections
