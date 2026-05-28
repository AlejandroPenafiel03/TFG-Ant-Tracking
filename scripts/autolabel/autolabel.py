# main.py
import cv2
import numpy as np
import os
from processor import process_frame   # Selecciona aquí el método deseado

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def mask_to_yolo(mask: np.ndarray, img_w: int, img_h: int, min_area: int = 50):
    """
    Convierte una máscara binaria en detecciones YOLOv8.
    Devuelve una lista de líneas YOLO (class cx cy w h).
    """
    detections = []

    # Filtrado morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Contornos
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)

        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h

        detections.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return detections


def process_video(video_path: str, output_dir: str, step: int = 15, min_area: int = 50):
    """
    Procesa un vídeo, genera detecciones YOLOv8 a partir de las máscaras
    y guarda imágenes RGB en images/ y etiquetas en labels/.
    """
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")

    ensure_dir(images_dir)
    ensure_dir(labels_dir)

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {video_path}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = process_frame(frame)

        if frame_idx % step == 0:
            h, w = frame.shape[:2]

            detections = mask_to_yolo(mask, w, h, min_area)

            # Guardar imagen RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_name = f"{video_name}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(images_dir, img_name),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            # Guardar etiquetas YOLO
            txt_name = f"{video_name}_frame_{frame_idx:06d}.txt"
            with open(os.path.join(labels_dir, txt_name), "w") as f:
                for line in detections:
                    f.write(line + "\n")

        frame_idx += 1

    cap.release()
    print(f"Procesado completado. Imágenes en {images_dir} y etiquetas en {labels_dir}")


if __name__ == "__main__":
    process_video(
        video_path="video.mp4",
        output_dir="output",
        step=15,
        min_area=50
    )
