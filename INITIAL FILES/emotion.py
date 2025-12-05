import cv2
import numpy as np
import os
import pandas as pd
import math


# -----------------------------------------------------------
#  Utility functions (geometry)
# -----------------------------------------------------------

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def mouth_openness(mouth_rect, face_h):
    (mx, my, mw, mh) = mouth_rect
    return mh / float(face_h)

def eyebrow_position(face_roi):
    h, w = face_roi.shape
    brow_region = face_roi[int(0.10*h):int(0.25*h), :]
    return np.mean(brow_region)


# -----------------------------------------------------------
#  Main Classical CV Emotion Analyzer (EXTENDED)
# -----------------------------------------------------------

def analyze_expression_extended(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "error", 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "no_face", 0

    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
    mouths = mouth_cascade.detectMultiScale(roi_gray, 1.7, 20)

    emotion = "neutral"
    score = 0

    # 1. Mouth / Smile detection
    if len(mouths) > 0:
        mouth_ratio = mouth_openness(mouths[0], h)

        if mouth_ratio > 0.25:
            emotion = "surprised"
            score = mouth_ratio * 3
        elif mouth_ratio > 0.18:
            emotion = "happy"
            score = mouth_ratio * 2

    # 2. Eye openness / tiredness
    if len(eyes) == 2:
        (ex, ey, ew, eh) = eyes[0]
        eye_ratio = eh / float(h)

        if eye_ratio < 0.10:
            emotion = "sad"
            score += 0.5
        elif eye_ratio > 0.18:
            emotion = "surprised"
            score += 1.0

    elif len(eyes) < 1:
        emotion = "sad"
        score += 0.6

    # 3. Eyebrow darkness → anger heuristic
    brow_value = eyebrow_position(roi_gray)
    if brow_value < 80:
        emotion = "angry"
        score += 0.7

    # 4. Head tilt
    center = x + w/2
    tilt = (center - img.shape[1]/2) / img.shape[1]
    if abs(tilt) > 0.15:
        emotion = "tilted_head"
        score += abs(tilt)

    return emotion, round(score, 3)


# -----------------------------------------------------------
#  Process Folder → CSV OUTPUT ONLY
# -----------------------------------------------------------

def process_folder(inputs_folder, output_csv):
    results = []

    for file in os.listdir(inputs_folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(inputs_folder, file)
            emotion, score = analyze_expression_extended(img_path)
            results.append([file, emotion, score])
            print(f"{file} → {emotion} (score: {score})")

    df = pd.DataFrame(results, columns=["File", "Emotion", "Score"])
    df.to_csv(output_csv, index=False)
    print("\nSaved:", output_csv)


# -----------------------------------------------------------
# Run
# -----------------------------------------------------------

if __name__ == "__main__":
    INPUT_FOLDER = "inputs"
    OUTPUT_CSV = "output/extended_cv_results.csv"

    os.makedirs("output", exist_ok=True)
    process_folder(INPUT_FOLDER, OUTPUT_CSV)
