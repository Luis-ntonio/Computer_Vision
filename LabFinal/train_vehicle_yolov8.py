from ultralytics import YOLO

# Inicia entrenamiento desde un modelo base YOLOv8
model = YOLO('yolov8n.pt')  # ultraligero para fine-tuning

model.train(
    data='./car-motorcycle-bus-truck-1/data.yaml',
    epochs=30,
    imgsz=640,
    batch=16,
    name='vehicle_yolov8_finetune'
)
