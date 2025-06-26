from roboflow import Roboflow
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Initialize Roboflow with your API key
API_ROBO = os.getenv("API_ROBO")

rf = Roboflow(api_key=API_ROBO)
datasets = os.getenv("DATASET_PLAYERS").strip().split(",")

for dataset in datasets:
    project = rf.workspace("lab-kycck").project(dataset)
    print("Project Name:", project.name)
    print("Project ID:", project)
    #project.generate_version(settings={"augmentation": {}, "preprocessing": {}})
    try:
        version = project.version(1)
    except:
        project.generate_version(settings={"augmentation": {}, "preprocessing": {}})
        version = project.version(1)
    
    version.export("yolov8")
    dataset = version.download("yolov8")  # crea carpeta con data.yaml, imágenes, labels
    print("Descargado en:", dataset.location)