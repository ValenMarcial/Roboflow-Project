import os
from roboflow import Roboflow
from ultralytics import RTDETR
from dotenv import load_dotenv
load_dotenv()

# --- CONFIGURACIÓN ---
API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = "vishal-sharma-hfh1k"
PROJECT = "fire-smoke-detection-1oemc"
VERSION = 3
                                     
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
        epochs=100,
        imgsz=640,      # BAJAR de 640 a 512 (Ahorra mucha VRAM)
        batch=4,        # BAJAR de 4 a 2 (Fundamental para la 1650)
        device=0,
        half=True,
        workers=8,      # Bajalo a 2 para no saturar el CPU
        project="runs/detect",
        name="mi_entrenamiento_rtdetr",
        exist_ok=True,
        save_dir="/home/valentin/Escritorio/Uni/Roboflow-Project/runs/detect/runs/detect/mi_entrenamiento_rtdetr",
    )
    
    print("\n✅ Entrenamiento finalizado.")
    print("Los pesos se han guardado en: runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    train_model()