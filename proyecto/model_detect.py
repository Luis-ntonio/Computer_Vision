from ultralytics import YOLO
import os
from dotenv import load_dotenv

#'list folders on ./data'
load_dotenv()
datasets = os.listdir('./data/detect')
print("Datasets disponibles:", datasets)

# Carga un modelo base pre-entrenado (para keypoints, úsalo si detectas keypoints; para deteción usa yolov8n.pt)
model = YOLO('yolo11m.pt')

# Entrena usando el dataset local
for dataset in datasets:
    data_yaml = f'./data/detect/{dataset}/data.yaml'
    if os.path.exists(data_yaml):
        print(f"Entrenando con el dataset: {dataset}")
    else:
        print(f"El archivo {data_yaml} no existe. Saltando el dataset.")
        continue

    # Entrena el modelo
    model.train(data=data_yaml, epochs=100, imgsz=640)
