import os
import av
import cv2
import numpy as np
import pandas as pd
import librosa

FRAME_RATE = 24
WINDOW = 30
AUDIO_SR = 22050

# ======================================
# AUDIO FEATURE EXTRACTOR
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

        intensity = np.sqrt(np.mean(win ** 2))
        spectral_centroid = librosa.feature.spectral_centroid(y=win, sr=sr).mean()
        spectral_flux = librosa.onset.onset_strength(y=win, sr=sr).mean()
        bandwidth = librosa.feature.spectral_bandwidth(y=win, sr=sr).mean()

        harmonicity = librosa.effects.harmonic(win).mean()
        onset_rate = librosa.onset.onset_detect(y=win, sr=sr).size
        zcr = librosa.feature.zero_crossing_rate(win).mean()

        pitches, mags = librosa.piptrack(y=win, sr=sr)
        pitch_vals = pitches[pitches > 0]
        pitch_mean = pitch_vals.mean() if len(pitch_vals) else 0
        pitch_var = pitch_vals.std() if len(pitch_vals) else 0

        # Simple reverb proxy
        reverb = np.abs(win[-1000:].mean())

        risk = 1 if (intensity > 0.15 or spectral_flux > 1.5) else 0

        results.append({
            "segment": seg + 1,
            "start_time": seg * WINDOW,
            "end_time": (seg + 1) * WINDOW,
            "audio_intensity": intensity,
            "audio_spectral_flux": spectral_flux,
            "audio_bandwidth": bandwidth,
            "audio_pitch_mean": pitch_mean,
            "audio_pitch_var": pitch_var,
            "audio_risk": risk
        })

    return results


# ======================================
# VIDEO FEATURE EXTRACTOR
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
    scene_changes = []

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
            scene_changes.append(1 if diff > 30 else 0)
        else:
            motion.append(0)
            scene_changes.append(0)

        prev_frame = gray

    total_secs = frame_count // FRAME_RATE
    total_segments = total_secs // WINDOW

    results = []

    for seg in range(total_segments):
        start_f = int(seg * WINDOW * FRAME_RATE)
        end_f = int((seg + 1) * WINDOW * FRAME_RATE)

        seg_bright = np.mean(brightness[start_f:end_f])
        seg_contrast = np.mean(contrast[start_f:end_f])
        seg_sat = np.mean(saturation[start_f:end_f])
        seg_motion = np.mean(motion[start_f:end_f])
        seg_scene = np.sum(scene_changes[start_f:end_f])

        risk = 1 if (seg_motion > 7 or seg_bright > 180) else 0

        results.append({
            "segment": seg + 1,
            "start_time": seg * WINDOW,
            "end_time": (seg + 1) * WINDOW,
            "video_brightness": seg_bright,
            "video_contrast": seg_contrast,
            "video_saturation": seg_sat,
            "video_motion": seg_motion,
            "video_scene_changes": seg_scene,
            "video_risk": risk
        })

    return results


# ======================================
# PROCESS FOLDERS (Independent)
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

    # Merge based on segment number
    df = pd.merge(df_audio, df_video, on=["segment", "start_time", "end_time"], how="outer")

    # =====================================
    # MULTIMODAL OVERLOAD INDEX
    # =====================================
    df["MCOI"] = df[["audio_risk", "video_risk"]].sum(axis=1)

    # =====================================
    # OVERLOAD TIMELINE PREDICTION
    # =====================================
    df = df.sort_values(by="segment")
    df["overload_prediction"] = (
        (df["MCOI"].shift(1) == 2) & (df["MCOI"] == 2)
    ).astype(int)

    df.to_csv(output_csv, index=False)
    print("\n[SAVED] :", output_csv)


# ======================================
# RUN
# ======================================
process_folders("audio_samples", "video_samples", "multimodal_overload_full.csv")
