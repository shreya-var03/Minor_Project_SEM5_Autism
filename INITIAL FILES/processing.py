import os
import cv2
import librosa
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# CONFIGURATION
# =========================================================
AUDIO_FOLDER = "audio_samples"
IMAGE_FOLDER = "image_samples"
VIDEO_FOLDER = "video_samples"
OUTPUT_CSV = "combined_sensory_overload_results.csv"

# ---- Feature Weights ----
AUDIO_WEIGHTS = {
    'intensity': 0.3,
    'onset_rate': 0.18,
    'spectral_flux': 0.18,
    'bandwidth': 0.1,
    'spectral_centroid': 0.08,
    'harmonicity': 0.06,
    'reverb': 0.05,
    'dynamic_range': 0.04,
    'pitch_variability': 0.01
}

IMAGE_WEIGHTS = {
    'brightness': 0.20,
    'brightness_change': 0.15,
    'contrast': 0.15,
    'saturation': 0.10,
    'color_variety': 0.10,
    'edge_density': 0.10,
    'motion': 0.10,
    'scene_change': 0.10
}

SOI_RISKY = 0.3
SOI_OVERLOAD = 0.6


# =========================================================
# AUDIO FEATURES (FAST MODE)
# =========================================================
def calculate_audio_features(y, sr):
    # process only 10 seconds, 22.05 kHz
    if len(y) > sr * 10:
        y = y[:sr * 10]

    rms = np.sqrt(np.mean(y**2))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_rate = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)) / (len(y)/sr)
    flux = np.mean(np.abs(np.diff(librosa.feature.rms(y=y)[0])))
    y_harm, y_perc = librosa.effects.hpss(y)
    hnr = np.mean(np.abs(y_harm)) / (np.mean(np.abs(y_perc)) + 1e-6)
    dynamic_range = np.ptp(librosa.feature.rms(y=y))
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitches_nonzero = pitches[pitches > 0]
    pitch_var = np.std(pitches_nonzero) if len(pitches_nonzero) > 0 else 0
    reverb = np.mean(np.abs(np.correlate(y[:sr*2], y[:sr*2], mode='full')))
    return {
        'intensity': rms,
        'spectral_centroid': centroid,
        'spectral_flux': flux,
        'bandwidth': bandwidth,
        'harmonicity': hnr,
        'onset_rate': onset_rate,
        'dynamic_range': dynamic_range,
        'pitch_variability': pitch_var,
        'reverb': reverb
    }

def compute_soi(features, weights):
    max_vals = {k: max(1.0, v) for k, v in features.items()}
    norm_features = {k: features[k] / max_vals[k] for k in features}
    return sum(weights.get(k, 0) * norm_features.get(k, 0) for k in weights)


# =========================================================
# IMAGE FEATURES (FAST)
# =========================================================
def calculate_image_features(frame, prev_frame=None):
    frame = cv2.resize(frame, (160, 120))  # smaller = faster
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    saturation = np.mean(hsv[:, :, 1])
    color_variety = np.var(hsv[:, :, 0])
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    brightness_change = motion = scene_change = 0
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(cv2.resize(prev_frame, (160, 120)), cv2.COLOR_BGR2GRAY)
        brightness_change = abs(np.mean(gray) - np.mean(prev_gray))
        diff = cv2.absdiff(gray, prev_gray)
        motion = np.mean(diff)
        scene_change = 1 - ssim(gray, prev_gray)

    return {
        'brightness': brightness,
        'brightness_change': brightness_change,
        'contrast': contrast,
        'saturation': saturation,
        'color_variety': color_variety,
        'edge_density': edge_density,
        'motion': motion,
        'scene_change': scene_change
    }

def compute_image_soi(features, weights):
    norm = {
        'brightness': features['brightness'] / 255,
        'brightness_change': min(features['brightness_change'] / 50, 1),
        'contrast': min(features['contrast'] / 80, 1),
        'saturation': features['saturation'] / 255,
        'color_variety': min(features['color_variety'] / 5000, 1),
        'edge_density': min(features['edge_density'] / 0.15, 1),
        'motion': min(features['motion'] / 50, 1),
        'scene_change': min(features['scene_change'] / 0.5, 1)
    }
    return sum(weights[k] * norm[k] for k in weights)


# =========================================================
# VIDEO ANALYSIS (FRAME SAMPLING)
# =========================================================
def analyze_video(video_path, frame_skip=15):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    all_soi = []
    frame_count = 0
    if not cap.isOpened():
        return None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            features = calculate_image_features(frame, prev_frame)
            soi = compute_image_soi(features, IMAGE_WEIGHTS)
            all_soi.append(soi)
            prev_frame = frame
        frame_count += 1
    cap.release()
    return np.mean(all_soi) if all_soi else None


# =========================================================
# MAIN PROCESSING FUNCTIONS
# =========================================================
def process_audio_file(f):
    path = os.path.join(AUDIO_FOLDER, f)
    try:
        y, sr = librosa.load(path, sr=22050, mono=True, duration=10)
        features = calculate_audio_features(y, sr)
        soi = compute_soi(features, AUDIO_WEIGHTS)
        classification = "Overload" if soi >= SOI_OVERLOAD else "Risky" if soi >= SOI_RISKY else "Calm"
        return {'type': 'audio', 'file': f, 'SOI': soi, 'classification': classification, **features}
    except Exception as e:
        print(f"[Audio error] {f}: {e}")
        return None

def process_image_file(f, prev_frame=[None]):
    path = os.path.join(IMAGE_FOLDER, f)
    frame = cv2.imread(path)
    if frame is None:
        return None
    features = calculate_image_features(frame, prev_frame[0])
    soi = compute_image_soi(features, IMAGE_WEIGHTS)
    classification = "Overload" if soi >= SOI_OVERLOAD else "Risky" if soi >= SOI_RISKY else "Calm"
    prev_frame[0] = frame
    return {'type': 'image', 'file': f, 'SOI': soi, 'classification': classification, **features}

def process_video_file(f):
    path = os.path.join(VIDEO_FOLDER, f)
    soi = analyze_video(path, frame_skip=15)
    if soi is None:
        return None
    classification = "Overload" if soi >= SOI_OVERLOAD else "Risky" if soi >= SOI_RISKY else "Calm"
    return {'type': 'video', 'file': f, 'SOI': soi, 'classification': classification}


# =========================================================
# RUN ALL TASKS IN PARALLEL
# =========================================================
results = []
audio_files = [f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith((".wav", ".mp3"))]
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

with ThreadPoolExecutor() as executor:
    futures = []
    for f in audio_files:
        futures.append(executor.submit(process_audio_file, f))
    for f in image_files:
        futures.append(executor.submit(process_image_file, f))
    for f in video_files:
        futures.append(executor.submit(process_video_file, f))

    for future in as_completed(futures):
        r = future.result()
        if r:
            results.append(r)

# =========================================================
# SAVE RESULTS
# =========================================================
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Combined results saved to: {OUTPUT_CSV}")
print(df.head())
