import cv2
from court_homography import CourtHomography
from people_ball_tracking import PeopleBallTracker

VIDEO = "./preprocessing/data/general/nba_20_min.mp4"
OUT_VIDEO = "output_annotated.mp4"
OUT_CSV = "trajectories.csv"

real_pts = [(0,0), (52*100,0), (52*100,94*100), (0,94*100)]
court = CourtHomography("/home/luis/Documents/UTEC/Computer_Vision/proyecto/models/runs/pose/train3/weights/best.pt", real_pts)

cap = cv2.VideoCapture(VIDEO)
ret, frame = cap.read()
H = court.compute(frame)
tracker = PeopleBallTracker("/home/luis/Documents/UTEC/Computer_Vision/proyecto/models/runs/detect/train3/weights/best.pt", H)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUT_VIDEO, fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frame_idx = 0
while ret:
    out_frame = tracker.process_frame(frame, frame_idx)
    out.write(out_frame)
    ret, frame = cap.read()
    frame_idx += 1

cap.release()
out.release()
tracker.save_csv(OUT_CSV)
print("✅ Pipeline completed.")
