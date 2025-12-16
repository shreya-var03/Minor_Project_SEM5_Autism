import os
import av
import cv2
import numpy as np
import pandas as pd
import librosa
import requests

# ===========================================================
# CONFIG
# ===========================================================
WINDOW = 10        # seconds per analysis window
STEP = 5           # seconds between windows
AUDIO_SR = 22050


# ===========================================================
# TIME HELPER
# ===========================================================
def to_timestamp(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}", h, m, s


# ===========================================================
# HAAR CASCADES
# ===========================================================
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
MOUTH_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)


# ===========================================================
# FACIAL EMOTION
# ===========================================================
def analyze_face_emotion_frame(img):
    if img is None:
        return "error", 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "no_face", 0.0

    (x, y, w, h) = faces[0]
    roi = gray[y:y+h, x:x+w]

    eyes = EYE_CASCADE.detectMultiScale(roi, 1.1, 10)
    mouths = MOUTH_CASCADE.detectMultiScale(roi, 1.7, 20)

    emotion = "neutral"
    score = 0.0

    if len(mouths) > 0:
        (_, _, _, mh) = mouths[0]
        ratio = mh / float(h)
        if ratio > 0.25:
            emotion, score = "surprised", ratio * 3
        elif ratio > 0.18:
            emotion, score = "happy", ratio * 2

    if len(eyes) < 1:
        emotion = "sad"
        score += 0.6

    brow = roi[int(0.10*h):int(0.25*h), :]
    if brow.size > 0 and brow.mean() < 80:
        emotion = "angry"
        score += 0.7

    return emotion, round(score, 3)


# ===========================================================
# AUDIO ANALYSIS (FROM VIDEO)
# ===========================================================
def analyze_audio(video_path, video_id):
    try:
        y, sr = librosa.load(video_path, sr=AUDIO_SR)
    except Exception:
        return []

    total_secs = len(y) // sr
    if total_secs < WINDOW:
        return []

    rows = []

    for start in range(0, total_secs - WINDOW + 1, STEP):
        win = y[start * sr:(start + WINDOW) * sr]
        ts, h, m, s = to_timestamp(start)

        intensity = float(np.sqrt(np.mean(win ** 2)))
        flux = float(librosa.onset.onset_strength(y=win, sr=sr).mean())
        bandwidth = float(librosa.feature.spectral_bandwidth(y=win, sr=sr).mean())

        rows.append({
            "video_id": video_id,
            "timestamp": ts,
            "hour": h,
            "minute": m,
            "second": s,
            "audio_intensity": intensity,
            "audio_flux": flux,
            "audio_bandwidth": bandwidth,
            "audio_risk": int(intensity > 0.15 or flux > 1.5)
        })

    return rows


# ===========================================================
# VIDEO ANALYSIS
# ===========================================================
def analyze_video(video_path, video_id):
    try:
        container = av.open(video_path)
    except Exception:
        return []

    stream = next((s for s in container.streams if s.type == "video"), None)
    if stream is None:
        return []

    fps = float(stream.average_rate) if stream.average_rate else 24.0

    motion, brightness = [], []
    emotions, scores = [], []
    prev_gray = None
    frames = 0

    for frame in container.decode(stream):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        brightness.append(gray.mean())
        motion.append(
            np.mean(cv2.absdiff(gray, prev_gray)) if prev_gray is not None else 0.0
        )
        prev_gray = gray

        e, s = analyze_face_emotion_frame(img)
        emotions.append(e)
        scores.append(s)
        frames += 1

    total_secs = int(frames / fps)
    if total_secs < WINDOW:
        return []

    rows = []
    win_frames = int(WINDOW * fps)

    for start in range(0, total_secs - WINDOW + 1, STEP):
        sf = int(start * fps)
        ef = sf + win_frames
        ts, h, m, s = to_timestamp(start)

        seg_motion = float(np.mean(motion[sf:ef]))
        seg_bright = float(np.mean(brightness[sf:ef]))

        candidates = [
            (e, sc) for e, sc in zip(emotions[sf:ef], scores[sf:ef])
            if e not in ("no_face", "error")
        ]

        face_e, face_s = ("no_face", 0.0)
        if candidates:
            face_e, face_s = max(candidates, key=lambda x: x[1])

        rows.append({
            "video_id": video_id,
            "timestamp": ts,
            "hour": h,
            "minute": m,
            "second": s,
            "video_motion": seg_motion,
            "video_brightness": seg_bright,
            "video_risk": int(seg_motion > 7 or seg_bright > 180),
            "face_emotion": face_e,
            "face_score": face_s
        })

    return rows


# ===========================================================
# WEATHER RISK
# ===========================================================
def get_weather_risk():
    try:
        r = requests.get(
            "http://api.weatherapi.com/v1/current.json",
            params={
                "key": "e9a4e51fca8541ea83e140443252111",
                "q": "Delhi",
                "aqi": "yes"
            }
        ).json()

        c = r["current"]
        aq = c.get("air_quality", {})

        return int(
            (c["temp_c"] > 35 or c["temp_c"] < 10) +
            (c["humidity"] > 75) +
            (c["uv"] >= 7) +
            (aq.get("pm2_5", 0) > 80)
        )
    except Exception:
        return 0


# ===========================================================
# MAIN PIPELINE
# ===========================================================
def process_multimodal(video_folder, output_csv=None):

    audio_rows, video_rows = [], []

    for f in os.listdir(video_folder):
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            path = os.path.join(video_folder, f)
            vid = os.path.splitext(f)[0]

            audio_rows.extend(analyze_audio(path, vid))
            video_rows.extend(analyze_video(path, vid))

    df_audio = pd.DataFrame(audio_rows)
    df_video = pd.DataFrame(video_rows)

    # ---- SAFE MERGE ----
    if not df_audio.empty and not df_video.empty:
        df = pd.merge(
            df_audio, df_video,
            on=["video_id", "timestamp", "hour", "minute", "second"],
            how="outer"
        )
    elif not df_audio.empty:
        df = df_audio.copy()
    elif not df_video.empty:
        df = df_video.copy()
    else:
        return None

    # ---- ENSURE RISK COLUMNS ----
    if "audio_risk" not in df.columns:
        df["audio_risk"] = 0
    else:
        df["audio_risk"] = df["audio_risk"].fillna(0).astype(int)

    if "video_risk" not in df.columns:
        df["video_risk"] = 0
    else:
        df["video_risk"] = df["video_risk"].fillna(0).astype(int)

    df["weather_risk"] = get_weather_risk()

    # ---- FINAL STATE ----
    df["total_risk"] = df["audio_risk"] + df["video_risk"] + df["weather_risk"]

    df["overall_state"] = df["total_risk"].apply(
        lambda x: "Severe Overload" if x >= 3
        else ("Moderate Overload" if x == 2 else "Calm")
    )

    df["overload_flag"] = (
        (df["overall_state"] == "Severe Overload") &
        (df.get("face_score", 0) >= 0.8)
    ).astype(int)

    df["distress_flag"] = (
        (df.get("face_score", 0) >= 0.6) &
        (df.get("face_emotion", "").isin(["sad", "angry", "surprised"]))
    ).astype(int)

    df["text_alert"] = df.apply(
        lambda r: "⚠️ Overload detected, intervention needed"
        if r["overload_flag"]
        else ("⚠️ Person appears distressed" if r["distress_flag"] else "No alert"),
        axis=1
    )

    if output_csv:
        df.to_csv(output_csv, index=False)

    return df
