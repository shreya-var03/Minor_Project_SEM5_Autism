import os
import av
import cv2
import numpy as np
import pandas as pd
import librosa
import math
import requests
import matplotlib.pyplot as plt

# ===========================================================
# CONFIG
# ===========================================================
FRAME_RATE = 24          # assumed FPS; adjust if needed
WINDOW = 10              # seconds per analysis window
STEP = 5                 # seconds between window starts (overlap)
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
# GLOBAL HAAR CASCADES (load once)
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
#  FACIAL EMOTION DETECTION ON FRAME
# ===========================================================
def analyze_face_emotion_frame(img_bgr):
    """Analyze a single BGR frame and return (emotion, score)."""
    if img_bgr is None:
        return "error", 0.0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return "no_face", 0.0

    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]

    eyes = EYE_CASCADE.detectMultiScale(roi_gray, 1.1, 10)
    mouths = MOUTH_CASCADE.detectMultiScale(roi_gray, 1.7, 20)

    emotion = "neutral"
    score = 0.0

    # ----- Mouth openness → happy / surprised -----
    if len(mouths) > 0:
        (mx, my, mw, mh) = mouths[0]
        mouth_ratio = mh / float(h)

        if mouth_ratio > 0.25:
            emotion = "surprised"
            score = mouth_ratio * 3
        elif mouth_ratio > 0.18:
            emotion = "happy"
            score = mouth_ratio * 2

    # ----- Eye closure → sad -----
    if len(eyes) < 1:
        emotion = "sad"
        score += 0.6

    # ----- Eyebrow region → angry -----
    brow_region = roi_gray[int(0.10*h):int(0.25*h), :]
    if brow_region.size > 0 and brow_region.mean() < 80:
        emotion = "angry"
        score += 0.7

    return emotion, round(score, 3)


# ===========================================================
#  AUDIO ANALYSIS (Librosa) - FROM VIDEO FILE
# ===========================================================
def analyze_audio(file_path, video_id):
    """
    Extracts audio from the given video and computes features on
    overlapping windows of length WINDOW seconds, step STEP seconds.
    """
    try:
        # librosa can load audio directly from many video files (mp4, mov, etc.)
        y, sr = librosa.load(file_path, sr=AUDIO_SR)
    except Exception as e:
        print("[ERROR] Cannot read audio from:", file_path, "|", e)
        return []

    samples_per_sec = sr
    win_samples = WINDOW * samples_per_sec

    total_secs = len(y) // samples_per_sec
    if total_secs < WINDOW:
        # Video too short for one full window
        return []

    results = []

    # Start a window every STEP seconds
    for start_sec in range(0, total_secs - WINDOW + 1, STEP):
        start_sample = start_sec * samples_per_sec
        end_sample = start_sample + win_samples
        win = y[start_sample:end_sample]

        timestamp, hh, mm, ss = to_timestamp(start_sec)

        intensity = np.sqrt(np.mean(win ** 2))
        spectral_flux = librosa.onset.onset_strength(y=win, sr=sr).mean()
        bandwidth = librosa.feature.spectral_bandwidth(y=win, sr=sr).mean()

        # simple rule, you can tune later
        risk = 1 if (intensity > 0.15 or spectral_flux > 1.5) else 0

        results.append({
            "video_id": video_id,
            "timestamp": timestamp,
            "hour": hh,
            "minute": mm,
            "second": ss,
            "audio_intensity": float(intensity),
            "audio_flux": float(spectral_flux),
            "audio_bandwidth": float(bandwidth),
            "audio_risk": int(risk)
        })

    return results


# ===========================================================
#  VIDEO ANALYSIS (Brightness / Motion / Saturation + FACES)
# ===========================================================
def analyze_video_with_faces(file_path, video_id):
    """
    Analyze video frames for brightness, contrast, saturation,
    motion and aggregate face emotions per overlapping window.
    """
    try:
        container = av.open(file_path)
    except Exception as e:
        print("[ERROR] Cannot read video:", file_path, "|", e)
        return []

    stream = next((s for s in container.streams if s.type == 'video'), None)
    if not stream:
        print("[WARN] No video stream found in:", file_path)
        return []

    stream.thread_type = "AUTO"

    brightness, contrast, saturation, motion = [], [], [], []
    frame_emotions, frame_scores = [], []
    prev_gray = None
    frame_count = 0

    # Decode all frames once
    for frame in container.decode(stream):
        frame_count += 1
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness.append(gray.mean())
        contrast.append(gray.std())

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation.append(hsv[:, :, 1].mean())

        if prev_gray is not None:
            diff = np.mean(cv2.absdiff(gray, prev_gray))
            motion.append(diff)
        else:
            motion.append(0.0)
        prev_gray = gray

        # Face emotion per frame
        emotion, score = analyze_face_emotion_frame(img)
        frame_emotions.append(emotion)
        frame_scores.append(score)

    if frame_count == 0:
        return []

    total_secs = frame_count // FRAME_RATE
    if total_secs < WINDOW:
        return []

    win_frames = WINDOW * FRAME_RATE

    results = []

    # start a window every STEP seconds
    for start_sec in range(0, total_secs - WINDOW + 1, STEP):
        timestamp, hh, mm, ss = to_timestamp(start_sec)

        start_f = int(start_sec * FRAME_RATE)
        end_f = start_f + win_frames
        if end_f > frame_count:
            break

        seg_bright = float(np.mean(brightness[start_f:end_f]))
        seg_contrast = float(np.mean(contrast[start_f:end_f]))
        seg_sat = float(np.mean(saturation[start_f:end_f]))
        seg_motion = float(np.mean(motion[start_f:end_f]))

        # simple video risk rule; tune later
        video_risk = 1 if (seg_motion > 7 or seg_bright > 180) else 0

        # ---- aggregate face emotion over frames in this window ----
        seg_emotions = frame_emotions[start_f:end_f]
        seg_scores = frame_scores[start_f:end_f]

        seg_emotion = "no_face"
        seg_score = 0.0

        candidates = [
            (e, s) for (e, s) in zip(seg_emotions, seg_scores)
            if e not in ("no_face", "error")
        ]
        if candidates:
            seg_emotion, seg_score = max(candidates, key=lambda x: x[1])

        results.append({
            "video_id": video_id,
            "timestamp": timestamp,
            "hour": hh,
            "minute": mm,
            "second": ss,
            "video_brightness": seg_bright,
            "video_contrast": seg_contrast,
            "video_saturation": seg_sat,
            "video_motion": seg_motion,
            "video_risk": int(video_risk),
            "face_emotion": seg_emotion,
            "face_score": float(seg_score)
        })

    return results


# ===========================================================
# WEATHER API + RISK ANALYSIS
# ===========================================================
def get_weather_data(city="Delhi"):
    # Replace with your own WeatherAPI.com key
    API_KEY = "e9a4e51fca8541ea83e140443252111"
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

    # Simple rules; tune as needed
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
# MAIN MULTIMODAL PIPELINE
# ===========================================================
def process_multimodal(video_folder, output_csv):

    audio_rows = []
    video_rows = []

    # ---------- PROCESS EACH VIDEO ----------
    for f in os.listdir(video_folder):
        if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
            video_path = os.path.join(video_folder, f)
            video_id = os.path.splitext(f)[0]

            print("[PROCESSING VIDEO]", video_path, "| video_id:", video_id)

            audio_rows.extend(analyze_audio(video_path, video_id))
            video_rows.extend(analyze_video_with_faces(video_path, video_id))

    df_audio = pd.DataFrame(audio_rows)
    df_video = pd.DataFrame(video_rows)

    # ---------- MERGE AUDIO + VIDEO ON video_id + TIME ----------
    if not df_audio.empty and not df_video.empty:
        df = pd.merge(
            df_audio,
            df_video,
            on=["video_id", "timestamp", "hour", "minute", "second"],
            how="outer"
        )
    elif not df_audio.empty:
        df = df_audio.copy()
    else:
        df = df_video.copy()

    if df.empty:
        print("[WARN] No data to save. Check videos / lengths.")
        df.to_csv(output_csv, index=False)
        print("[SAVED EMPTY CSV]:", output_csv)
        return

    df = df.sort_values(by=["video_id", "hour", "minute", "second"])

    # Ensure risk columns exist
    if "audio_risk" not in df.columns:
        df["audio_risk"] = 0
    else:
        df["audio_risk"] = df["audio_risk"].fillna(0)

    if "video_risk" not in df.columns:
        df["video_risk"] = 0
    else:
        df["video_risk"] = df["video_risk"].fillna(0)

    # Ensure face columns exist
    if "face_emotion" not in df.columns:
        df["face_emotion"] = "no_face"
    if "face_score" not in df.columns:
        df["face_score"] = 0.0

    # ---------- WEATHER INTEGRATION ----------
    weather = get_weather_data()

    if weather:
        weather_risk = compute_weather_risk(weather)
        print("[WEATHER]", weather, "| Risk =", weather_risk)
    else:
        weather = {
            "temp": None, "humidity": None,
            "uv": None, "condition": None,
            "pm25": None
        }
        weather_risk = 0

    df["weather_temp"] = weather["temp"]
    df["weather_humidity"] = weather["humidity"]
    df["weather_uv"] = weather["uv"]
    df["weather_condition"] = weather["condition"]
    df["weather_pm25"] = weather["pm25"]
    df["weather_risk"] = weather_risk

    # ---------- OVERALL RISK & STATE ----------
    df["total_risk"] = (
        df["audio_risk"].astype(int)
        + df["video_risk"].astype(int)
        + df["weather_risk"].astype(int)
    )

    df["overall_state"] = df["total_risk"].apply(
        lambda x: "Severe Overload" if x >= 3
        else ("Moderate Overload" if x == 2 else "Calm")
    )

    # Optional: mark segments where any face is present
    df["has_face"] = df["face_emotion"].apply(
        lambda e: 0 if e in ("no_face", "error", None) else 1
    )

    # ---------- ALERT FLAGS ----------
    def compute_overload_flag(row):
        """
        Overload = environment is severe AND facial expression is strong.
        Threshold 0.8 can be tuned.
        """
        if row["overall_state"] == "Severe Overload" and row["face_score"] >= 0.8:
            return 1
        return 0

    def compute_distress_flag(row):
        """
        Distress = person looks emotionally negative even if not in full overload.
        Conditions (tune as needed):
          - Moderate overload + some facial intensity, OR
          - Strong 'sad' / 'angry' / 'surprised' face even when overall_state is Calm.
        """
        negative_emotions = {"sad", "angry", "surprised"}

        if row["face_emotion"] in negative_emotions and row["face_score"] >= 0.6:
            return 1

        if row["overall_state"] == "Moderate Overload" and row["face_score"] >= 0.5:
            return 1

        return 0

    df["overload_flag"] = df.apply(compute_overload_flag, axis=1)
    df["distress_flag"] = df.apply(compute_distress_flag, axis=1)

    # ---------- TEXT ALERT COLUMN ----------
    def compute_text_alert(row):
        if row["overload_flag"] == 1:
            return "⚠️ Overload detected, intervention needed"
        if row["distress_flag"] == 1:
            return "⚠️ Person appears distressed, monitor closely"
        return "No alert"

    df["text_alert"] = df.apply(compute_text_alert, axis=1)

    df.to_csv(output_csv, index=False)
    print("\n[SAVED with WEATHER]:", output_csv)


# ===========================================================
# PLOTTING: OVERALL STATE + ALERTS OVER TIME FOR ONE VIDEO
# ===========================================================
def plot_overall_state_over_time(csv_path, video_id):
    """
    Load the CSV, filter for one video_id, and plot:
      - overall_state vs time
      - highlight distress and overload segments with different markers/colors.
    """
    df = pd.read_csv(csv_path)

    # keep only this video's rows
    df = df[df["video_id"] == video_id].copy()
    if df.empty:
        print(f"[WARN] No rows found for video_id = {video_id}")
        return

    # convert timestamp (hh:mm:ss) to total seconds (for x-axis)
    def time_to_seconds(t):
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)

    df["time_sec"] = df["timestamp"].apply(time_to_seconds)

    # map overall_state to numeric levels for plotting
    state_to_level = {"Calm": 0, "Moderate Overload": 1, "Severe Overload": 2}
    df["state_level"] = df["overall_state"].map(state_to_level)

    plt.figure()

    # --- base line plot of overall_state ---
    plt.plot(df["time_sec"], df["state_level"], marker="o", linestyle="-", label="Overall State")

    # --- highlight overload segments (red triangles) ---
    overload_points = df[df["overload_flag"] == 1]
    if not overload_points.empty:
        plt.scatter(
            overload_points["time_sec"],
            overload_points["state_level"],
            marker="^",
            color="red",
            s=100,
            label="Overload alert"
        )

    # --- highlight distress segments (orange squares) that are not overload ---
    distress_points = df[(df["distress_flag"] == 1) & (df["overload_flag"] == 0)]
    if not distress_points.empty:
        plt.scatter(
            distress_points["time_sec"],
            distress_points["state_level"],
            marker="s",
            color="orange",
            s=80,
            label="Distress alert"
        )

    plt.yticks(
        [0, 1, 2],
        ["Calm", "Moderate Overload", "Severe Overload"]
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Overall State")
    plt.title(f"Overall State & Alerts Over Time - video_id = {video_id}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===========================================================
# RUN PIPELINE
# ===========================================================
if __name__ == "__main__":
    # Run analysis
    process_multimodal(
        video_folder="video_samples",
        output_csv="combined_multimodal.csv"
    )

    # Example: plot for one specific video (use filename without extension)
    # plot_overall_state_over_time("combined_multimodal.csv", video_id="your_video_name_here")
