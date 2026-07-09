import cv2
import numpy as np
import mediapipe as mp

mp_face = mp.solutions.face_detection


def extract_face_crop(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extracts a single padded face crop from a still image.
    Used by the image analysis endpoint to pass face-only region to heuristics.
    Returns None if no face is detected or crop is too small.
    """
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        results = fd.process(img_rgb)
        if not results.detections:
            return None

        detection = results.detections[0]  # take first/best face
        box = detection.location_data.relative_bounding_box
        h, w, _ = img_rgb.shape

        x1 = int(box.xmin * w)
        y1 = int(box.ymin * h)
        x2 = int((box.xmin + box.width) * w)
        y2 = int((box.ymin + box.height) * h)

        pad = 0.4
        w_box = x2 - x1
        h_box = y2 - y1
        nx1 = max(0, int(x1 - w_box * pad))
        ny1 = max(0, int(y1 - h_box * pad))
        nx2 = min(w, int(x2 + w_box * pad))
        ny2 = min(h, int(y2 + h_box * pad))

        crop = img_rgb[ny1:ny2, nx1:nx2]
        return crop if crop.shape[0] >= 80 and crop.shape[1] >= 80 else None


def extract_faces_from_video(video_path: str, maxFrames: int = 40) -> list:
    """
    Extracts padded face crops from evenly sampled frames in a video.
    Returns a list of RGB numpy arrays (face crops).
    """
    cap = cv2.VideoCapture(video_path)
    faces = []
    extracted_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, total_frames // maxFrames)

    current_frame_idx = 0

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        while True:
            ret = cap.grab()
            if not ret or extracted_count >= maxFrames:
                break

            if current_frame_idx % stride == 0:
                ret, frame = cap.retrieve()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = fd.process(frame_rgb)

                if results.detections:
                    for detection in results.detections:
                        box = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x1 = int(box.xmin * w)
                        y1 = int(box.ymin * h)
                        x2 = int((box.xmin + box.width) * w)
                        y2 = int((box.ymin + box.height) * h)

                        pad = 0.4
                        w_box = x2 - x1
                        h_box = y2 - y1
                        nx1 = max(0, int(x1 - w_box * pad))
                        ny1 = max(0, int(y1 - h_box * pad))
                        nx2 = min(w, int(x2 + w_box * pad))
                        ny2 = min(h, int(y2 + h_box * pad))

                        face = frame_rgb[ny1:ny2, nx1:nx2]
                        if face.shape[0] < 80 or face.shape[1] < 80:
                            continue
                        if face.size > 0:
                            faces.append(face)
                            extracted_count += 1

            current_frame_idx += 1  # FIX: was outside loop before, never incremented

    cap.release()
    return faces