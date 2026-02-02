import cv2
import supervision as sv
from ultralytics import RTDETR
import os

# --- CONFIGURACIÓN ---

video_name = input("Ingrese el nombre del video (con extensión, ej: video.mp4): ")

WEIGHTS_PATH = "best.pt"             # Tu modelo entrenado por tu amigo
SOURCE_VIDEO_PATH = os.path.join(os.path.dirname(__file__), video_name) # <--- CAMBIÁ ESTO por el nombre de tu video
OUTPUT_VIDEO_PATH = "video_resultado.mp4"
CONFIDENCE = 0.6                      # Confianza (ajustala si ves mucho o poco)

def process_video():
    # 1. Cargar modelo
    print(f"Cargando modelo desde: {WEIGHTS_PATH}")
    model = RTDETR(WEIGHTS_PATH)

    # 2. Configurar video de entrada
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return

    # Obtener propiedades del video original para guardar el nuevo igual
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    # 3. Configurar guardado de video (Codecs)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    # 4. Configuradores de anotación
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    print("🎥 Procesando video... (Apretá 'q' para salir antes)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break # Se terminó el video

        # --- INFERENCIA ---
        results = model.predict(frame, conf=CONFIDENCE, verbose=False)
        
        # --- DIBUJAR ---
        detections = sv.Detections.from_ultralytics(results[0])

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(detections.data['class_name'], detections.confidence)
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # --- GUARDAR Y MOSTRAR ---
        out.write(frame)       # Guardar en disco
        cv2.imshow("RT-DETR Video", frame) # Mostrar en pantalla

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Video guardado en: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    process_video()