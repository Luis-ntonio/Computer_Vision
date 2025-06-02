import numpy as np
import math

def associate_detections_to_trackers(detections, trackers, dist_threshold=50):
    """
    detections: lista de (x, y, w, h, conf, class_id)
    trackers:   lista de instancias de KalmanTracker
    dist_threshold: umbral máximo de distancia para hacer match

    Devuelve:
      matches       : lista de pares (idx_tracker, idx_detection)
      unmatched_det : indices de detecciones sin emparejar
      unmatched_trk : indices de trackers sin emparejar
    """
    if len(trackers) == 0:
        return [], list(range(len(detections))), []

    # 1) Centros de las detecciones
    det_centers = []
    for det in detections:
        x, y, w, h, _, _ = det
        det_centers.append((x + w / 2.0, y + h / 2.0))

    # 2) Centros predichos por cada tracker
    trk_centers = []
    for trk in trackers:
        cx, cy = trk.predict()
        trk_centers.append((cx, cy))

    # 3) Matriz de distancias (trackers × detecciones)
    dist_matrix = np.zeros((len(trk_centers), len(det_centers)), dtype=np.float32)
    for t, (tcx, tcy) in enumerate(trk_centers):
        for d, (dcx, dcy) in enumerate(det_centers):
            dist_matrix[t, d] = math.hypot(tcx - dcx, tcy - dcy)

    # 4) Generar todas las parejas (dist, t, d) y ordenarlas
    all_pairs = []
    for t in range(len(trk_centers)):
        for d in range(len(det_centers)):
            all_pairs.append((dist_matrix[t, d], t, d))
    all_pairs.sort(key=lambda x: x[0])

    matches = []
    taken_trk = set()
    taken_det = set()
    for dist, t, d in all_pairs:
        if dist > dist_threshold:
            continue
        if t in taken_trk or d in taken_det:
            continue
        taken_trk.add(t)
        taken_det.add(d)
        matches.append((t, d))

    unmatched_trk = [t for t in range(len(trackers)) if t not in taken_trk]
    unmatched_det = [d for d in range(len(detections)) if d not in taken_det]
    return matches, unmatched_det, unmatched_trk