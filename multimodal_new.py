import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import cv2

from scipy.io import wavfile
from scipy.signal import find_peaks
from PIL import Image

# =========================================================
# CONFIG
# =========================================================

AUDIO_FOLDER = "audio_samples"
RESULTS_FOLDER = "results"
PHYSIO_CSV = os.path.join("inputs", "physio_data.csv")  # change if needed
IMAGE_PATH = "img.jpg"  # image for visual classification

# ================== WEATHER API ==================
API_KEY = "5b4ce54a13604a8795e105427252108"  # your WeatherAPI key
CITY = "Delhi"


def get_weather():
    """Fetch current weather + AQI; return dict and pretty string."""
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={CITY}&aqi=yes"
    try:
        response = requests.get(url, timeout=5).json()
    except Exception as e:
        print("[Weather] Error fetching data:", e)
        return None, "Weather: unavailable"

    if "current" not in response:
        print("[Weather] Invalid response:", response)
        return None, "Weather: unavailable"

    current = response["current"]
    temp = current["temp_c"]
    humidity = current["humidity"]
    uv = current["uv"]
    condition = current["condition"]["text"]
    air_quality = current.get("air_quality", {})
    pm25 = air_quality.get("pm2_5", 0)

    w = {
        "temp": temp,
        "humidity": humidity,
        "uv": uv,
        "pm25": pm25,
        "condition": condition,
    }
    info = (
        f"Weather: {condition}, Temp: {temp}°C, "
        f"Humidity: {humidity}%, UV: {uv}, PM2.5: {pm25}"
    )
    return w, info


# ================== AUDIO ==================

def get_sound_db_from_file(filepath: str) -> float:
    """Read WAV file and return average dB level."""
    rate, data = wavfile.read(filepath)

    # Stereo -> mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    data = data.astype(float)
    rms = np.sqrt(np.mean(data ** 2))
    db = 20 * np.log10(rms + 1e-6)
    return round(db, 2)


def classify_audio(sound_db: float, weather: dict | None):
    status = "Calm"
    reasons = []

    temp = weather.get("temp") if weather else None
    humidity = weather.get("humidity") if weather else None
    uv = weather.get("uv") if weather else None
    pm25 = weather.get("pm25") if weather else None

    # Weather stress rules
    if temp is not None and temp > 32:
        reasons.append("High temperature")
    if humidity is not None and humidity > 70:
        reasons.append("High humidity")
    if uv is not None and uv > 7:
        reasons.append("Strong UV / sunlight")
    if pm25 is not None and pm25 > 50:
        reasons.append("Poor air quality")

    # Noise thresholds
    if sound_db > 85:
        reasons.append("Extreme noise")
        status = "Overload"
    elif sound_db > 70:
        reasons.append("High noise")
        status = "Risky"

    # If any weather risks but not already overload
    if reasons and status != "Overload":
        status = "Risky"

    return status, reasons


# ================== VISUAL ==================

def load_image_from_file(path: str):
    """Load local image and return as numpy array (or None)."""
    if not os.path.exists(path):
        print(f"[Visual] Image not found: {path}")
        return None
    img = Image.open(path).convert("RGB")
    return np.array(img)


def classify_visual_features(img_array: np.ndarray):
    """Classify image as calm/crowded and estimate color tone."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # 1. Edge density (busyness)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # 2. Variance (textures/objects)
    variance = np.var(gray)

    # 3. Brightness
    brightness = np.mean(gray)

    # Decide crowd vs calm (simple rules; tweak if needed)
    if edge_density > 0.10 or variance > 5000:
        visual_status = "Crowded"
        visual_reasons = (
            f"High edge density ({edge_density:.3f}) and/or variance "
            f"({variance:.1f}) → visually busy"
        )
    else:
        visual_status = "Calm"
        visual_reasons = (
            f"Low edge density ({edge_density:.3f}) and variance "
            f"({variance:.1f}) → visually simple"
        )

    # Color tone
    avg_color = img_array.mean(axis=(0, 1))  # [R, G, B]
    if avg_color[2] > avg_color[1] and avg_color[2] > avg_color[0]:
        color_tone = "Warm (reddish)"
    elif avg_color[0] > avg_color[1]:
        color_tone = "Cool (bluish)"
    else:
        color_tone = "Neutral"

    extra_info = (
        f"Brightness: {brightness:.1f}, Var: {variance:.1f}, "
        f"Edge density: {edge_density:.3f}"
    )

    return visual_status, visual_reasons, color_tone, extra_info


# ================== PHYSIO / GAZE CSV ==================

def summarize_physio_state(csv_path: str):
    """
    Read Engagnition-style CSV and classify physio/gaze state.
    Very generic: tries common column names and uses simple heuristics.
    """
    if not os.path.exists(csv_path):
        print(f"[Physio] CSV not found: {csv_path}")
        return "Not available", ["Physio CSV file not found"]

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print("[Physio] Cannot read CSV:", e)
        return "Unknown", ["Could not read physio CSV"]

    reasons = []
    score = 0  # higher = more overload risk

    def get_col(possible_names):
        for name in possible_names:
            if name in df.columns:
                return df[name].astype(float)
        return None

    hr = get_col(["HR", "HeartRate", "heart_rate", "bpm"])
    hrv = get_col(["HRV", "hrv", "SDNN", "sdnn"])
    eda = get_col(["EDA", "eda", "GSR", "gsr"])
    eng = get_col(["Engagement", "engagement", "EngagementIndex", "engagement_index"])

    # Heart rate
    if hr is not None:
        mean_hr = hr.mean()
        if mean_hr > 100:
            score += 1
            reasons.append(f"High average heart rate ({mean_hr:.1f} bpm)")
        elif mean_hr < 50:
            reasons.append(f"Low average heart rate ({mean_hr:.1f} bpm)")
        else:
            reasons.append(f"Heart rate in moderate range ({mean_hr:.1f} bpm)")

    # HRV
    if hrv is not None:
        mean_hrv = hrv.mean()
        if mean_hrv < 30:  # ms, rough stress threshold
            score += 1
            reasons.append(f"Low HRV ({mean_hrv:.1f} ms) → reduced variability / stress")
        else:
            reasons.append(f"HRV not critically low ({mean_hrv:.1f} ms)")

    # EDA / GSR
    if eda is not None:
        eda_series = eda.fillna(method="ffill")
        thr = eda_series.mean() + eda_series.std()
        peaks, _ = find_peaks(eda_series, height=thr)
        peak_ratio = len(peaks) / max(1, len(eda_series))
        if peak_ratio > 0.02:  # >2% of samples are peaks
            score += 1
            reasons.append(
                f"Frequent EDA peaks (approx. {peak_ratio*100:.1f}% of samples) → high arousal"
            )
        else:
            reasons.append(
                f"EDA peaks not very frequent ({peak_ratio*100:.1f}% of samples)"
            )

    # Engagement (optional)
    if eng is not None:
        mean_eng = eng.mean()
        if mean_eng < 0.3:
            score += 1
            reasons.append(f"Low average engagement ({mean_eng:.2f})")
        else:
            reasons.append(f"Engagement not extremely low ({mean_eng:.2f})")

    # Final state
    if score >= 2:
        state = "Overload"
    elif score == 1:
        state = "Risky"
    else:
        state = "Calm"

    if hr is None and hrv is None and eda is None and eng is None:
        state = "Unknown"
        reasons = ["No recognizable physio columns found in CSV"]

    return state, reasons


# ================== MAIN ==================

def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # Weather
    weather, weather_info = get_weather()
    print(weather_info)

    # Visual
    img_array = load_image_from_file(IMAGE_PATH)
    if img_array is not None:
        (
            visual_status,
            visual_reasons,
            color_tone,
            visual_extra,
        ) = classify_visual_features(img_array)
    else:
        visual_status = "Not available"
        visual_reasons = "Image not found"
        color_tone = "Unknown"
        visual_extra = ""

    # Physio
    physio_state, physio_reasons = summarize_physio_state(PHYSIO_CSV)

    print("\n=== Processing audio files ===\n")
    count = 1

    if not os.path.exists(AUDIO_FOLDER):
        print(f"[Audio] Folder not found: {AUDIO_FOLDER}")
        return

    for filename in os.listdir(AUDIO_FOLDER):
        if not filename.lower().endswith(".wav"):
            continue

        filepath = os.path.join(AUDIO_FOLDER, filename)

        # AUDIO
        try:
            sound_db = get_sound_db_from_file(filepath)
        except Exception as e:
            print(f"[Audio] Error reading {filename}:", e)
            continue

        audio_status, audio_reasons = classify_audio(sound_db, weather)

        # Build report text
        report = []
        report.append("========== MULTIMODAL ENVIRONMENT REPORT ==========")
        report.append(f"File: {filename}")
        report.append("")

        report.append(">> WEATHER INFO")
        report.append(weather_info)
        report.append("")

        report.append(">> AUDIO CLASSIFICATION")
        report.append(f"Sound Level: {sound_db} dB")
        report.append(f"Status: {audio_status}")
        if audio_reasons:
            report.append("Reasons: " + ", ".join(audio_reasons))
        report.append("")

        report.append(">> VISUAL CLASSIFICATION")
        report.append(f"Status: {visual_status}")
        report.append(f"Color Tone: {color_tone}")
        if visual_reasons:
            report.append("Details: " + visual_reasons)
        if visual_extra:
            report.append("Extra: " + visual_extra)
        report.append("")

        report.append(">> PHYSIO / GAZE CLASSIFICATION")
        report.append(f"Overall Physio State: {physio_state}")
        if physio_reasons:
            report.append("Reasons:")
            for r in physio_reasons:
                report.append(f"  - {r}")

        report.append("===================================================")

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outpath = os.path.join(
            RESULTS_FOLDER, f"report_{count}_{timestamp}.txt"
        )
        with open(outpath, "w", encoding="utf-8") as f:
            f.write("\n".join(report))

        print(f"✔ Report {count} saved → {outpath}")
        count += 1

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
