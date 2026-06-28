# video_utils.py
import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_detection

def extract_faces_from_video(path, max_frames=40):
    cap = cv2.VideoCapture(path)
    faces = []
    count = 0

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fd.process(frame_rgb)

            if results.detections:
                for det in results.detections:
                    box = det.location_data.relative_bounding_box
                    h, w, _ = frame_rgb.shape

                    x1 = int(box.xmin * w)
                    y1 = int(box.ymin * h)
                    x2 = x1 + int(box.width * w)
                    y2 = y1 + int(box.height * h)

                    pad = 0.4  # 40% expansion
                    w_box = x2 - x1
                    h_box = y2 - y1

                    nx1 = max(0, int(x1 - w_box * pad))
                    ny1 = max(0, int(y1 - h_box * pad))
                    nx2 = min(w, int(x2 + w_box * pad))
                    ny2 = min(h, int(y2 + h_box * pad))
                    face = frame_rgb[ny1:ny2, nx1:nx2]

                    if face.shape[0] < 180 or face.shape[1] < 180:
                        continue

                    if face.size > 0:
                        faces.append(face)

            count += 1
            if count >= max_frames:
                break

    cap.release()
    return faces