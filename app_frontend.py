import os
import av
import cv2
import numpy as np
import pandas as pd
import librosa
import gradio as gr
import requests   # << ADDED for weather API

# ===========================================================
# CONFIG
# ===========================================================
FRAME_RATE = 24
WINDOW = 30          # seconds per segment
AUDIO_SR = 22050

# ===========================================================
# WEATHER API FUNCTION
# ===========================================================
def get_weather():
    API_KEY = "5b4ce54a13604a8795e105427252108"
    CITY = "Delhi"

    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={CITY}&aqi=yes"
    try:
        response = requests.get(url).json()
    except:
        return "Could not fetch WeatherAPI response."

    if "current" not in response:
        return f"WeatherAPI Error: {response}"

    current = response["current"]
    temp = current["temp_c"]
    humidity = current["humidity"]
    uv = current["uv"]
    condition = current["condition"]["text"]
    air_quality = current.get("air_quality", {})
    pm25 = air_quality.get("pm2_5", 0)

    weather_info = (
        f"Weather: **{condition}**, "
        f"Temp: **{temp}°C**, "
        f"Humidity: **{humidity}%**, "
        f"UV Index: **{uv}**, "
        f"PM2.5: **{pm25}**"
    )
    return weather_info


# ===========================================================
# TIMESTAMP HELPER
# ===========================================================
def to_timestamp(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}", h, m, s


# ===========================================================
# AUDIO ANALYSIS
# ===========================================================
def analyze_audio(file_path):
    if file_path is None:
        return []

    try:
        y, sr = librosa.load(file_path, sr=AUDIO_SR)
    except Exception as e:
        print("[ERROR] Cannot read audio:", file_path, e)
        return []

    samples_per_win = sr * WINDOW
    total_segments = max(1, len(y) // samples_per_win)
    results = []

    for seg in range(total_segments):
        start = seg * samples_per_win
        end = min((seg + 1) * samples_per_win, len(y))
        win = y[start:end]

        start_sec = start / sr
        timestamp, hh, mm, ss = to_timestamp(start_sec)

        intensity = float(np.sqrt(np.mean(win ** 2)))
        spectral_flux = float(librosa.onset.onset_strength(y=win, sr=sr).mean())
        bandwidth = float(librosa.feature.spectral_bandwidth(y=win, sr=sr).mean())

        risk = 1 if (intensity > 0.15 or spectral_flux > 1.5) else 0

        results.append({
            "timestamp": timestamp,
            "hour": hh,
            "minute": mm,
            "second": ss,
            "audio_intensity": intensity,
            "audio_flux": spectral_flux,
            "audio_bandwidth": bandwidth,
            "audio_risk": risk
        })

    return results


# ===========================================================
# VIDEO ANALYSIS
# ===========================================================
def analyze_video(file_path):
    if file_path is None:
        return []

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

    total_secs = frame_count / FRAME_RATE
    total_segments = max(1, int(total_secs // WINDOW))

    results = []

    for seg in range(total_segments):
        start_sec = seg * WINDOW
        timestamp, hh, mm, ss = to_timestamp(start_sec)

        start_f = int(seg * WINDOW * FRAME_RATE)
        end_f = int(min((seg + 1) * WINDOW * FRAME_RATE, frame_count))

        seg_bright = float(np.mean(brightness[start_f:end_f]))
        seg_contrast = float(np.mean(contrast[start_f:end_f]))
        seg_sat = float(np.mean(saturation[start_f:end_f]))
        seg_motion = float(np.mean(motion[start_f:end_f]))

        risk = 1 if (seg_motion > 7 or seg_bright > 180) else 0

        results.append({
            "timestamp": timestamp,
            "hour": hh,
            "minute": mm,
            "second": ss,
            "video_brightness": seg_bright,
            "video_contrast": seg_contrast,
            "video_saturation": seg_sat,
            "video_motion": seg_motion,
            "video_risk": risk
        })

    return results


# ===========================================================
# FACIAL EMOTION
# ===========================================================
def mouth_openness(mouth_rect, face_h):
    (mx, my, mw, mh) = mouth_rect
    return mh / float(face_h)

def eyebrow_position(face_roi):
    h, w = face_roi.shape
    brow_region = face_roi[int(0.10*h):int(0.25*h), :]
    return np.mean(brow_region)

def analyze_face_emotion(image_path):
    if image_path is None:
        return "no_image", 0.0

    img = cv2.imread(image_path)
    if img is None:
        return "error", 0.0

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
        return "no_face", 0.0

    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
    mouths = mouth_cascade.detectMultiScale(roi_gray, 1.7, 20)

    emotion = "neutral"
    score = 0.0

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
# PIPELINE FOR FRONTEND
# ===========================================================
def _get_path(x):
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("name") or x.get("filepath") or x.get("path")
    return None


def run_multimodal(audio_file, video_file, image_file):
    audio_path = _get_path(audio_file)
    video_path = _get_path(video_file)
    image_path = _get_path(image_file)

    # Weather data
    weather_info = get_weather()

    audio_rows = analyze_audio(audio_path) if audio_path else []
    video_rows = analyze_video(video_path) if video_path else []
    emotion, emo_score = analyze_face_emotion(image_path)

    df_audio = pd.DataFrame(audio_rows)
    df_video = pd.DataFrame(video_rows)

    if not df_audio.empty or not df_video.empty:
        if df_audio.empty:
            df = df_video.copy()
        elif df_video.empty:
            df = df_audio.copy()
        else:
            df = pd.merge(df_audio, df_video,
                          on=["timestamp", "hour", "minute", "second"],
                          how="outer")

        df = df.sort_values(by=["hour", "minute", "second"])

        df["audio_risk"] = df.get("audio_risk", 0).fillna(0)
        df["video_risk"] = df.get("video_risk", 0).fillna(0)

        df["MCOI"] = df["audio_risk"] + df["video_risk"]
        df["state"] = df["MCOI"].apply(lambda x: "Risky" if x >= 2 else "Calm")

        overall_state = df["state"].iloc[-1]
    else:
        df = pd.DataFrame()
        overall_state = "No data"

    # Final summary block
    lines = []
    lines.append("## 🌤️ Weather Data")
    lines.append(weather_info)
    lines.append("")
    lines.append("## 🎧🎥 Multimodal Analysis Result")
    lines.append(f"- **Overall Environment State**: `{overall_state}`")
    lines.append(f"- **Face Emotion**: `{emotion}` (score: `{emo_score}`)")
    lines.append("")
    lines.append("### Segment Summary")
    lines.append(f"- Audio segments analysed: `{len(audio_rows)}`")
    lines.append(f"- Video segments analysed: `{len(video_rows)}`")

    return "\n".join(lines), df, weather_info


# ===========================================================
# GRADIO UI
# ===========================================================
with gr.Blocks() as demo:
    gr.Markdown("# 🎧🎥 Multimodal Sensory Environment Analyzer + 🌤️ Weather API")
    gr.Markdown("Upload audio, video & face image. Weather data is fetched automatically for Delhi.")

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone", "upload"],
                               type="filepath", label="Audio Input")
        video_input = gr.Video(sources=["webcam", "upload"],
                               label="Video Input")
        image_input = gr.Image(sources=["webcam", "upload"],
                               type="filepath", label="Face Image")

    run_btn = gr.Button("Run Full Analysis")

    summary_out = gr.Markdown(label="Summary Output")
    table_out = gr.Dataframe(label="Segment-wise Multimodal Table")
    weather_out = gr.Markdown(label="Live Weather Info")

    run_btn.click(
        fn=run_multimodal,
        inputs=[audio_input, video_input, image_input],
        outputs=[summary_out, table_out, weather_out]
    )

if __name__ == "__main__":
    demo.launch(share=True)
