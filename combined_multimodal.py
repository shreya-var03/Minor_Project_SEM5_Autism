import os
import av
import cv2
import numpy as np
import pandas as pd
import librosa
import math
import requests

# ===========================================================
# CONFIG
# ===========================================================
FRAME_RATE = 24
WINDOW = 30          # seconds per segment
AUDIO_SR = 22050


# ===========================================================
# TIMESTAMP HELPER
# ===========================================================
def to_timestamp(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}", h, m, s


# ===========================================================
#  AUDIO ANALYSIS (Librosa)
# ===========================================================
def analyze_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=AUDIO_SR)
    except Exception as e:
        print("[ERROR] Cannot read audio:", file_path, e)
        return []

    samples_per_win = sr * WINDOW
    if samples_per_win <= 0:
        return []

    total_segments = len(y) // samples_per_win
    if total_segments == 0:
        total_segments = 1  # handle short audio

    results = []

    for seg in range(total_segments):
        start = seg * samples_per_win
        end = min((seg + 1) * samples_per_win, len(y))
        win = y[start:end]
        if len(win) == 0:
            continue

        start_sec = seg * WINDOW
        timestamp, hh, mm, ss = to_timestamp(start_sec)

        intensity = float(np.sqrt(np.mean(win ** 2)))
        spectral_flux = float(librosa.onset.onset_strength(y=win, sr=sr).mean())
        bandwidth = float(librosa.feature.spectral_bandwidth(y=win, sr=sr).mean())

        risk = 1 if (intensity > 0.15 or spectral_flux > 1.5) else 0

        results.append({
            "timestamp": timestamp, "hour": hh, "minute": mm, "second": ss,
            "audio_intensity": intensity,
            "audio_flux": spectral_flux,
            "audio_bandwidth": bandwidth,
            "audio_risk": risk
        })

    return results


# ===========================================================
#  VIDEO ANALYSIS (Brightness / Motion / Saturation)
# ===========================================================
def analyze_video(file_path):
    try:
        container = av.open(file_path)
    except Exception as e:
        print("[ERROR] Cannot read video:", file_path, e)
        return []

    stream = next((s for s in container.streams if s.type == 'video'), None)
    if not stream:
        return []

    stream.thread_type = "AUTO"

    brightness, contrast, saturation, motion = [], [], [], []
    prev_frame = None
    frame_count = 0

    for frame in container.decode(stream):
        frame_count += 1
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness.append(gray.mean())
        contrast.append(gray.std())

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation.append(hsv[:, :, 1].mean())

        if prev_frame is not None:
            diff = np.mean(cv2.absdiff(gray, prev_frame))
            motion.append(diff)
        else:
            motion.append(0)

        prev_frame = gray

    if frame_count == 0:
        return []

    total_secs = max(1, frame_count // FRAME_RATE)
    total_segments = max(1, total_secs // WINDOW) if total_secs >= WINDOW else 1

    results = []

    for seg in range(total_segments):
        start_sec = seg * WINDOW
        timestamp, hh, mm, ss = to_timestamp(start_sec)

        start_f = int(seg * WINDOW * FRAME_RATE)
        end_f = int(min((seg + 1) * WINDOW * FRAME_RATE, frame_count))
        if start_f >= end_f:
            continue

        seg_bright = float(np.mean(brightness[start_f:end_f]))
        seg_contrast = float(np.mean(contrast[start_f:end_f]))
        seg_sat = float(np.mean(saturation[start_f:end_f]))
        seg_motion = float(np.mean(motion[start_f:end_f]))

        risk = 1 if (seg_motion > 7 or seg_bright > 180) else 0

        results.append({
            "timestamp": timestamp, "hour": hh, "minute": mm, "second": ss,
            "video_brightness": seg_bright,
            "video_contrast": seg_contrast,
            "video_saturation": seg_sat,
            "video_motion": seg_motion,
            "video_risk": risk
        })

    return results


# ===========================================================
# FACIAL EMOTION DETECTION (Classical CV)
# ===========================================================
def mouth_openness(mouth_rect, face_h):
    (mx, my, mw, mh) = mouth_rect
    return mh / float(face_h)

def eyebrow_position(face_roi):
    h, w = face_roi.shape
    brow_region = face_roi[int(0.10*h):int(0.25*h), :]
    return np.mean(brow_region)

def analyze_face_emotion(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "error", 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    mouth_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_smile.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return "no_face", 0

    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
    mouths = mouth_cascade.detectMultiScale(roi_gray, 1.7, 20)

    emotion = "neutral"
    score = 0

    if len(mouths) > 0:
        mouth_ratio = mouth_openness(mouths[0], h)

        if mouth_ratio > 0.25:
            emotion = "surprised"
            score = mouth_ratio * 3
        elif mouth_ratio > 0.18:
            emotion = "happy"
            score = mouth_ratio * 2

    if len(eyes) < 1:
        emotion = "sad"
        score += 0.6

    brow_value = eyebrow_position(roi_gray)
    if brow_value < 80:
        emotion = "angry"
        score += 0.7

    return emotion, round(score, 3)


# ===========================================================
# WEATHER API + RISK ANALYSIS
# ===========================================================
def get_weather_data(city="Delhi"):
    API_KEY = "e9a4e51fca8541ea83e140443252111"  # your WeatherAPI key
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=yes"

    try:
        response = requests.get(url).json()
    except Exception as e:
        print("[ERROR] Weather API failed:", e)
        return None

    if "current" not in response:
        print("[ERROR] Invalid weather response:", response)
        return None

    current = response["current"]
    air_quality = current.get("air_quality", {})

    weather = {
        "temp": current["temp_c"],
        "humidity": current["humidity"],
        "uv": current["uv"],
        "condition": current["condition"]["text"],
        "pm25": air_quality.get("pm2_5", 0)
    }

    return weather


def compute_weather_risk(w):
    risk = 0
    if w["temp"] > 35 or w["temp"] < 10:
        risk += 1
    if w["humidity"] > 75:
        risk += 1
    if w["uv"] >= 7:
        risk += 1
    if w["pm25"] > 80:
        risk += 1
    return risk


# ===========================================================
# BATCH PIPELINE (folders -> CSV)  -- optional, you can still keep it
# ===========================================================
def process_multimodal(audio_folder, video_folder, image_folder, output_csv):
    audio_rows = []
    for f in os.listdir(audio_folder):
        if f.lower().endswith(".mp3"):
            audio_rows.extend(analyze_audio(os.path.join(audio_folder, f)))

    video_rows = []
    for f in os.listdir(video_folder):
        if f.lower().endswith((".mp4", ".mov", ".webm")):
            video_rows.extend(analyze_video(os.path.join(video_folder, f)))

    image_rows = []
    for f in os.listdir(image_folder):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            emotion, score = analyze_face_emotion(os.path.join(image_folder, f))
            image_rows.append({
                "timestamp": f.split(".")[0],
                "face_emotion": emotion,
                "face_score": score
            })

    df_audio = pd.DataFrame(audio_rows)
    df_video = pd.DataFrame(video_rows)
    df_faces = pd.DataFrame(image_rows)

    df = pd.merge(df_audio, df_video,
                  on=["timestamp", "hour", "minute", "second"],
                  how="outer")
    df = pd.merge(df, df_faces, on="timestamp", how="left")

    df = df.sort_values(by=["hour", "minute", "second"])
    df["audio_risk"] = df["audio_risk"].fillna(0)
    df["video_risk"] = df["video_risk"].fillna(0)

    weather = get_weather_data()
    if weather:
        weather_risk = compute_weather_risk(weather)
        print("[WEATHER]", weather, "| Risk =", weather_risk)
    else:
        weather = {"temp": None, "humidity": None, "uv": None,
                   "condition": None, "pm25": None}
        weather_risk = 0

    df["weather_temp"] = weather["temp"]
    df["weather_humidity"] = weather["humidity"]
    df["weather_uv"] = weather["uv"]
    df["weather_condition"] = weather["condition"]
    df["weather_pm25"] = weather["pm25"]
    df["weather_risk"] = weather_risk

    df["total_risk"] = (
        df["audio_risk"].astype(int)
        + df["video_risk"].astype(int)
        + df["weather_risk"].astype(int)
    )

    df["overall_state"] = df["total_risk"].apply(
        lambda x: "Severe Overload" if x >= 3
        else ("Moderate Overload" if x == 2 else "Calm")
    )

    df.to_csv(output_csv, index=False)
    print("\n[SAVED with WEATHER]:", output_csv)


# ===========================================================
# GRADIO-FRIENDLY PIPELINE (single audio/video/image)
# ===========================================================
def run_multimodal(audio_file, video_file, image_file):
    """
    This is what Gradio will call.
    audio_file, video_file, image_file: filepaths (strings) or None.
    Returns: (summary_markdown, dataframe, weather_markdown)
    """

    # ---------- AUDIO ----------
    audio_rows = []
    if audio_file:
        try:
            audio_rows = analyze_audio(audio_file)
        except Exception as e:
            print("[ERROR] run_multimodal audio:", e)

    # ---------- VIDEO ----------
    video_rows = []
    if video_file:
        try:
            video_rows = analyze_video(video_file)
        except Exception as e:
            print("[ERROR] run_multimodal video:", e)

    # ---------- FACE / IMAGE ----------
    emotion = "no_image"
    face_score = 0
    if image_file:
        try:
            emotion, face_score = analyze_face_emotion(image_file)
        except Exception as e:
            print("[ERROR] run_multimodal image:", e)
            emotion, face_score = "error", 0

    df_audio = pd.DataFrame(audio_rows)
    df_video = pd.DataFrame(video_rows)

    # merge logic robust to missing audio/video
    if not df_audio.empty and not df_video.empty:
        df = pd.merge(
            df_audio,
            df_video,
            on=["timestamp", "hour", "minute", "second"],
            how="outer"
        )
    elif not df_audio.empty:
        df = df_audio.copy()
    elif not df_video.empty:
        df = df_video.copy()
    else:
        df = pd.DataFrame(columns=["timestamp", "hour", "minute", "second"])

    # guarantee risk columns
    if "audio_risk" in df.columns:
        df["audio_risk"] = df["audio_risk"].fillna(0)
    else:
        df["audio_risk"] = 0

    if "video_risk" in df.columns:
        df["video_risk"] = df["video_risk"].fillna(0)
    else:
        df["video_risk"] = 0

    # ---------- WEATHER ----------
    # ---------- WEATHER ----------
    weather = get_weather_data("Delhi")
    if weather:
        weather_risk = compute_weather_risk(weather)

        if weather_risk >= 3:
            weather_state = "Severe weather risk"
        elif weather_risk == 2:
            weather_state = "Moderate weather risk"
        elif weather_risk == 1:
            weather_state = "Mild weather risk"
        else:
            weather_state = "Safe / low weather risk"

        weather_md = f"**Weather Risk Level:** {weather_state} (score: {weather_risk})"
    else:
        weather_risk = 0
        weather_md = "**Weather Risk Level:** Not available (API error)"

    # ---------- Risk merge ----------
    if not df.empty:
        df = df.sort_values(by=["hour", "minute", "second"])
        df["weather_risk"] = weather_risk
        df["total_risk"] = (
            df["audio_risk"].astype(int)
            + df["video_risk"].astype(int)
            + df["weather_risk"].astype(int)
        )
        df["overall_state"] = df["total_risk"].apply(
            lambda x: "Severe Overload" if x >= 3
            else ("Moderate Overload" if x == 2 else "Calm")
        )
        overall_state = df["overall_state"].iloc[-1]
    else:
        overall_state = "No data"

    # ---------- SUMMARY ----------
    lines = []
    lines.append("## 🌤️ Weather Data")
    lines.append(weather_md)
    lines.append("")
    lines.append("## 🎧🎥 Multimodal Environment Analysis")
    lines.append(f"- **Overall Environment State**: `{overall_state}`")
    lines.append(f"- **Face Emotion**: `{emotion}` (score: `{face_score}`)")
    lines.append("")
    lines.append("### Segment Summary")
    lines.append(f"- Audio segments analysed: `{len(audio_rows)}`")
    lines.append(f"- Video segments analysed: `{len(video_rows)}`")

    summary_text = "\n".join(lines)

    return summary_text, df, weather_md


# ===========================================================
# CLI ENTRY POINT (optional batch mode)
# ===========================================================
if __name__ == "__main__":
    process_multimodal(
        audio_folder="audio_samples",
        video_folder="video_samples",
        image_folder="inputs",
        output_csv="combined_multimodal.csv"
    )
