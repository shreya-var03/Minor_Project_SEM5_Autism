import os
import av
import cv2
import numpy as np
import pandas as pd
import librosa

FRAME_RATE = 24
WINDOW = 30
AUDIO_SR = 22050

# Convert seconds → HH:MM:SS
def to_timestamp(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}", h, m, s


# ======================================
# AUDIO FEATURE EXTRACTOR (Real timestamps)
# ======================================
def analyze_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=AUDIO_SR)
    except:
        print("[ERROR] Cannot read audio:", file_path)
        return []

    samples_per_win = sr * WINDOW
    total_segments = len(y) // samples_per_win
    results = []

    for seg in range(total_segments):
        start = seg * samples_per_win
        end = (seg + 1) * samples_per_win
        win = y[start:end]

        start_sec = seg * WINDOW
        end_sec = (seg + 1) * WINDOW

        timestamp, hh, mm, ss = to_timestamp(start_sec)

        intensity = np.sqrt(np.mean(win ** 2))
        spectral_flux = librosa.onset.onset_strength(y=win, sr=sr).mean()
        bandwidth = librosa.feature.spectral_bandwidth(y=win, sr=sr).mean()

        risk = 1 if (intensity > 0.15 or spectral_flux > 1.5) else 0

        results.append({
            "file": os.path.basename(file_path),
            "timestamp": timestamp,
            "hour": hh, "minute": mm, "second": ss,
            "audio_intensity": intensity,
            "audio_flux": spectral_flux,
            "audio_bandwidth": bandwidth,
            "audio_risk": risk
        })

    return results


# ======================================
# VIDEO FEATURE EXTRACTOR (Real timestamps)
# ======================================
def analyze_video(file_path):
    try:
        container = av.open(file_path)
    except:
        print("[ERROR] Cannot read video:", file_path)
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

    total_secs = frame_count // FRAME_RATE
    total_segments = total_secs // WINDOW

    results = []

    for seg in range(total_segments):
        start_sec = seg * WINDOW
        end_sec = (seg + 1) * WINDOW

        timestamp, hh, mm, ss = to_timestamp(start_sec)

        start_f = int(seg * WINDOW * FRAME_RATE)
        end_f = int((seg + 1) * WINDOW * FRAME_RATE)

        seg_bright = np.mean(brightness[start_f:end_f])
        seg_contrast = np.mean(contrast[start_f:end_f])
        seg_sat = np.mean(saturation[start_f:end_f])
        seg_motion = np.mean(motion[start_f:end_f])

        risk = 1 if (seg_motion > 7 or seg_bright > 180) else 0

        results.append({
            "file": os.path.basename(file_path),
            "timestamp": timestamp,
            "hour": hh, "minute": mm, "second": ss,
            "video_brightness": seg_bright,
            "video_contrast": seg_contrast,
            "video_saturation": seg_sat,
            "video_motion": seg_motion,
            "video_risk": risk
        })

    return results


# ======================================
# PREDICT NEXT TIMESTAMP STATE
# ======================================
def predict_state(prev, current):
    if prev == "Risky" and current == "Calm":
        return "Likely Calm"
    if prev == "Calm" and current == "Risky":
        return "Likely Risky"
    return current


# Final environment output
def final_environment(states):
    last = states[-3:] if len(states) >= 3 else states
    if last.count("Risky") >= 2:
        return "Environment Likely Overloading"
    return "Environment Likely Safe"


# ======================================
# PROCESS FOLDERS (Independent Processing)
# ======================================
def process_folders(audio_folder, video_folder, output_csv):

    audio_rows = []
    for f in os.listdir(audio_folder):
        if f.lower().endswith(".mp3"):
            audio_rows.extend(analyze_audio(os.path.join(audio_folder, f)))

    video_rows = []
    for f in os.listdir(video_folder):
        if f.lower().endswith((".mp4", ".mov")):
            video_rows.extend(analyze_video(os.path.join(video_folder, f)))

    df_audio = pd.DataFrame(audio_rows)
    df_video = pd.DataFrame(video_rows)

    # Merge using timestamp
    df = pd.merge(df_audio, df_video,
                  on=["timestamp", "hour", "minute", "second"],
                  how="outer")

    df = df.sort_values(by=["hour", "minute", "second"])

    # Combined multimodal index
    df["audio_risk"] = df["audio_risk"].fillna(0)
    df["video_risk"] = df["video_risk"].fillna(0)

    df["MCOI"] = df["audio_risk"] + df["video_risk"]

    # State
    df["state"] = df["MCOI"].apply(lambda x: "Risky" if x >= 2 else "Calm")

    # Prediction
    predictions = []
    prev = "Calm"

    for s in df["state"]:
        predictions.append(predict_state(prev, s))
        prev = s

    df["prediction"] = predictions

    # Save CSV
    df.to_csv(output_csv, index=False)
    print("\n[SAVED] :", output_csv)

    print("\nFINAL ENVIRONMENT OUTCOME →", final_environment(list(df["state"])))


# ======================================
# RUN
# ======================================
process_folders("audio_samples", "video_samples", "multimodal_overload_full.csv")
