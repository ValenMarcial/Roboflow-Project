import os
from roboflow import Roboflow
from ultralytics import RTDETR
from dotenv import load_dotenv
load_dotenv()

# --- CONFIGURACIÓN ---
API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = "valentin-kguz0"
PROJECT = "deteccion-incendio-forestal"
VERSION = 1


def train_model():
    print("Descargando dataset...")
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    
    # IMPORTANTE: RF-DETR prefiere el formato COCO
    dataset = project.version(VERSION).download("yolov12")

    print("Cargando arquitectura RF-DETR...")
    # Puedes usar 'rf-detr-small' para velocidad o 'rf-detr-large' para precisión
    model = RTDETR("rtdetr-l.pt")

    print("🔥 Iniciando entrenamiento en GPU...")
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=1,           # Solo 1 época para testear errores
        imgsz=640,
        batch=2,            # Batch bajo para que no falle por memoria
        device=0,
        name="test_merge"
    )
    
    print("\n✅ Entrenamiento finalizado.")
    print("Los pesos se han guardado en: runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    train_model()