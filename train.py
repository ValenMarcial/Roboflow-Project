import os
from roboflow import Roboflow
from ultralytics import RTDETR

# --- CONFIGURACIÓN ---
API_KEY = "TU_API_KEY_DE_ROBOFLOW"
WORKSPACE = "tu-workspace"
PROJECT = "tu-project-name"
VERSION = 1

def train_model():
    print("Descargando dataset...")
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    
    # IMPORTANTE: RF-DETR prefiere el formato COCO
    dataset = project.version(VERSION).download("coco")

    print("Cargando arquitectura RF-DETR...")
    # Puedes usar 'rf-detr-small' para velocidad o 'rf-detr-large' para precisión
    model = RTDETR("rf-detr-large")

    print("Iniciando entrenamiento...")
    # Los resultados se guardarán automáticamente en la carpeta 'runs/'
    model.train(
        data=dataset.location,
        epochs=50,          # Ajusta según necesidad
        imgsz=640,        # Tamaño de imagen (640 es estándar)
        batch=2,            # Como base se toma 8 pero bajar a 4 si te da error de memoria (CUDA OOM)
        device="cpu",           # Usar con GPU es recomendable device=0
        val=True            # Validar durante el entrenamiento
    )
    
    print("\n✅ Entrenamiento finalizado.")
    print("Los pesos se han guardado en: runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    train_model()