import numpy as np
import math
# ——————————————
# Función de asociación detecciones ↔ trackers (flehmen
# exactamente igual que antes)
# ——————————————
def associate_detections_to_trackers(detections, trackers, dist_threshold=50):
    if len(trackers) == 0:
        return [], list(range(len(detections))), []

    det_centers = []
    for det in detections:
        x, y, w, h, _, _ = det
        det_centers.append((x + w / 2., y + h / 2.))

    trk_centers = []
    for trk in trackers:
        cx, cy = trk.predict()
        trk_centers.append((cx, cy))

    dist_matrix = np.zeros((len(trk_centers), len(det_centers)), dtype=np.float32)
    for t, (tcx, tcy) in enumerate(trk_centers):
        for d, (dcx, dcy) in enumerate(det_centers):
            dist_matrix[t, d] = math.hypot(tcx - dcx, tcy - dcy)

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