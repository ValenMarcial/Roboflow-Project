import cv2
import supervision as sv
from ultralytics import RTDETR

# --- CONFIGURACIÓN ---
WEIGHTS_PATH = "runs/detect/runs/detect/mi_entrenamiento_rtdetr/weights/best.pt" 
IMAGE_PATH = "imagen_prueba.jpg"  # Pon aquí la foto que quieras probar
CONFIDENCE = 0.5                  # Umbral de confianza (50%)

def run_inference():
    # 1. Cargar el modelo personalizado
    print(f"Cargando pesos desde: {WEIGHTS_PATH}")
    try:
        model = RTDETR(WEIGHTS_PATH)
    except Exception as e:
        print(f"Error: No se encuentra el archivo de pesos \n{e}")
        return

    # 2. Cargar imagen
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("Error: No se encontró la imagen de prueba.")
        return

    # 3. Predicción (Inferencia)
    print("Analizando imagen...")
    results = model.predict(source=image, conf=CONFIDENCE)
    
    # 4. Procesar resultados para dibujar (Usando Supervision)
    # RF-DETR devuelve una lista de resultados, tomamos el primero [0]
    detections = sv.Detections.from_ultralytics(results[0])

    # 5. Configurar anotadores (Cajas y Etiquetas)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    # Crear etiquetas con Nombre de clase + Confianza
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections.data['class_name'], detections.confidence)
    ]

    # 6. Dibujar sobre la imagen
    annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # 7. Mostrar y Guardar
    output_name = "resultado_rfdetr.jpg"
    cv2.imwrite(output_name, annotated_image)
    print(f"✅ Resultado guardado como: {output_name}")
    
    # Mostrar en ventana
    cv2.imshow("RF-DETR Resultado", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()