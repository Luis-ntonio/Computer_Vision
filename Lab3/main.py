import cv2
from kallman_filter import KalmanTracker
from yolo_model import detect_objects_ultralytics
from util import associate_detections_to_trackers


# ——————————————
# Main loop: captura video, detecta, asocia, trackea
# ——————————————
if __name__ == "__main__":
    # 1) Abre video (0 = webcam, o reemplaza con ruta a archivo)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara/video.")
        exit()

    trackers = []
    max_age_frames = 10
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # 3) Detecta objetos con Ultralytics
        detections_all = detect_objects_ultralytics(frame, conf_threshold=0.25)

        # (Opcional) Filtrado por clase, p. ej. solo 'person' (cls == 0 en COCO)
        detections = []
        for det in detections_all:
            x, y, w, h, conf, cls_id = det
            # si solo quieres personas:
            # if cls_id != 0: continue
            detections.append(det)

        # 4) Asocia detecciones ⇄ trackers
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, trackers, dist_threshold=50)

        # 5) Actualiza trackers emparejados
        for trk_idx, det_idx in matches:
            bbox = detections[det_idx][:4]
            trackers[trk_idx].update(bbox)

        # 6) Crea nuevos trackers para detecciones no emparejadas
        for idx in unmatched_dets:
            bbox = detections[idx][:4]
            new_trk = KalmanTracker(bbox)
            trackers.append(new_trk)

        # 7) Elimina trackers “viejos”
        to_del = []
        for i, trk in enumerate(trackers):
            if trk.time_since_update > max_age_frames:
                to_del.append(i)
        for i in sorted(to_del, reverse=True):
            del trackers[i]

        # 8) Dibuja boxes predichas (filtradas) y raw detections
        for trk in trackers:
            x, y, w, h = trk.get_state()
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {trk.id}", (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Dibuja detecciones crudas en azul (opcional)
        for det in detections:
            x, y, w, h, _, _ = det
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 1)

        cv2.putText(frame, f"Frame: {frame_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("YOLOv8 + Kalman Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
