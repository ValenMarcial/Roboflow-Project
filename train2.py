import os
from rfdetr import RFDETRBase
from roboflow import Roboflow

API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = "vishal-sharma-hfh1k"
PROJECT = "fire-smoke-detection-1oemc"
VERSION = 3

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)

# IMPORTANTE: RF-DETR prefiere el formato COCO
dataset = project.version(VERSION).download("yolov12")

model = RFDETRBase()

model.train(
    dataset_dir=f"{dataset.location}/data.yaml",
    epochs=30,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="/home/valentin/Escritorio/Uni/Roboflow-Project/rf-dert/runs/detect/runs/detect/mi_entrenamiento_rfdetr"
    #resume=dir es un path para guardar un checkpoint y continuar el entrenamiento
)