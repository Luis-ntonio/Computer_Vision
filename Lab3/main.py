import cv2
from kallman_filter import KalmanTracker
from yolo_model import detect_objects_ultralytics
from util import associate_detections_to_trackers
from dotenv import load_dotenv
import os

load_dotenv

num_future = int(os.getenv("NUM_FUTURE"))

# ——————————————
# Main loop: captura video, detecta, asocia, trackea
# ——————————————
if __name__ == "__main__":
    # 4.2) Abrir fuente de vídeo (0 = webcam; si no funciona, sustituir por 'video.mp4')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara/video. Verifica que exista y esté libre.")
        exit()

    trackers = []            # Lista de instancias KalmanTracker activas
    max_age_frames = 10      # Si un tracker no se actualiza en 10 frames, lo eliminamos
    window_name = "Seguimiento + Proyección"  # Nombre único de ventana
    frame_id = 0

    # Crear una única ventana antes del bucle
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            # Si no conseguimos leer un frame, rompemos el bucle
            break

        frame_id += 1

        # ————— 4.3) Detectar objetos en este frame —————
        detections_all = detect_objects_ultralytics(frame, conf_threshold=0.25)

        # (Opcional) Filtrar por clase, e.g., solo 'person' (cls_id == 0)
        detections = []
        for det in detections_all:
            x, y, w, h, conf, cls_id = det
            if cls_id != 0:
                 continue
            detections.append(det)

        # ————— 4.4) Asociar detecciones ⇄ trackers —————
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, trackers, dist_threshold=50)

        # ————— 4.5) Actualizar trackers emparejados —————
        for trk_idx, det_idx in matches:
            bbox = detections[det_idx][:4]  # (x, y, w, h)
            trackers[trk_idx].update(bbox)

        # ————— 4.6) Crear nuevos trackers para detecciones no emparejadas —————
        for idx in unmatched_dets:
            bbox = detections[idx][:4]
            new_trk = KalmanTracker(bbox)
            trackers.append(new_trk)

        # ————— 4.7) Eliminar trackers “viejos” —————
        to_delete = []
        for i, trk in enumerate(trackers):
            if trk.time_since_update > max_age_frames:
                to_delete.append(i)
        for i in sorted(to_delete, reverse=True):
            del trackers[i]

        # ————— 4.8) Dibujar cajas y proyección futura —————
        for trk in trackers:
            # 4.8.1) Caja estimada (verde)
            x, y, w, h = trk.get_state()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {trk.id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 4.8.2) Obtener proyección futura para los próximos `num_future` frames
            futuros = trk.predict_future(num_future)
            puntos = [(int(cx_f), int(cy_f)) for cx_f, cy_f in futuros]

            # 4.8.3) Dibujar línea continua entre cada par de puntos futuros (amarillo)
            for j in range(len(puntos) - 1):
                cv2.line(frame, puntos[j], puntos[j + 1], (0, 255, 255), 2, lineType=cv2.LINE_AA)

            # 4.8.4) Dibujar un pequeño círculo en cada punto proyectado
            for (cx_f, cy_f) in puntos:
                cv2.circle(frame, (cx_f, cy_f), 4, (0, 255, 255), -1)

        # ————— 4.9) (Opcional) Dibujar detecciones “raw” en azul —————
        for det in detections:
            x, y, w, h, _, _ = det
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

        cv2.putText(frame, f"Frame: {frame_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # 4.10) Mostrar el frame resultante en la ventana única
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Presiona 'q' para salir
            break

    # ————— 5) Al salir, liberar recursos y cerrar ventana —————
    cap.release()
    cv2.destroyAllWindows()