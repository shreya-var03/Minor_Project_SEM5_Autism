import numpy as np
import requests
from PIL import Image
import cv2
import os
from scipy.io import wavfile
from datetime import datetime

# ================== WEATHER API ==================
API_KEY = "5b4ce54a13604a8795e105427252108"  # Replace with your WeatherAPI.com key
CITY = "Delhi"
url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={CITY}&aqi=yes"
response = requests.get(url).json()

if "current" not in response:
    print("Error fetching weather data:", response)
    exit()

current = response["current"]
temp = current["temp_c"]
humidity = current["humidity"]
uv = current["uv"]
condition = current["condition"]["text"]
air_quality = current.get("air_quality", {})
pm25 = air_quality.get("pm2_5", 0)

weather_info = f"Weather: {condition}, Temp: {temp}°C, Humidity: {humidity}%, UV: {uv}, PM2.5: {pm25}"


# ================== WEATHER CLASSIFICATION ==================
def classify_weather(temp, humidity, uv, pm25):
    reasons = []

    if temp > 32:
        reasons.append("High Temperature")
    if humidity > 70:
        reasons.append("High Humidity")
    if uv > 7:
        reasons.append("Strong UV / Sunlight")
    if pm25 > 50:
        reasons.append("Poor Air Quality")

    # Decide weather status
    if len(reasons) == 0:
        status = "Calm"
    elif len(reasons) <= 2:
        status = "Risky"
    else:
        status = "Overload"

    return status, reasons


# ================== AUDIO CLASSIFICATION ==================
def get_sound_db_from_file(filepath):
    """Read WAV file and return average dB level"""
    rate, data = wavfile.read(filepath)

    # Stereo -> mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    rms = np.sqrt(np.mean(data**2))
    db = 20 * np.log10(rms + 1e-6)
    return round(db, 2)


def classify_audio(sound_db):
    status = "Calm"
    reasons = []

    if sound_db > 85:
        reasons.append("Extreme Noise")
        status = "Overload"
    elif sound_db > 70:
        reasons.append("High Noise")
        status = "Risky"

    return status, reasons


# ================== VISUAL CLASSIFICATION ==================
def load_image_from_file(path):
    """Load local image and return as numpy array"""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def classify_visual_features(img_array):
    """Classify image as calm/crowded based on features"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # 1. Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / edges.size

    # 2. Variance (high variance means lots of textures/objects)
    variance = np.var(gray)

    # 3. Brightness
    brightness = np.mean(gray)

    # Decide crowd vs calm
    if edge_density > 20 or variance > 5000:  # tweak thresholds
        visual_status = "Crowded"
        visual_reasons = f"High edge density ({edge_density:.2f}) and variance ({variance:.2f}) → looks busy"
    else:
        visual_status = "Calm"
        visual_reasons = f"Low edge density ({edge_density:.2f}) and variance ({variance:.2f}) → looks calm"

    # Color tone
    avg_color = img_array.mean(axis=(0, 1))
    if avg_color[2] > avg_color[1] and avg_color[2] > avg_color[0]:
        color_tone = "Warm (reddish)"
    elif avg_color[0] > avg_color[1]:
        color_tone = "Cool (bluish)"
    else:
        color_tone = "Neutral"

    return visual_status, visual_reasons, color_tone


# ================== OVERALL ENVIRONMENT SCORE ==================
WEIGHTS = {
    "audio": 0.5,   # strongest effect
    "visual": 0.3,
    "weather": 0.2
}

def compute_environment_score(audio_status, visual_status, weather_status):
    mapping = {"Calm": 0, "Risky": 1, "Overload": 2, "Crowded": 1}

    audio_score = mapping.get(audio_status, 0)
    visual_score = mapping.get(visual_status, 0)
    weather_score = mapping.get(weather_status, 0)

    total_score = (
        audio_score * WEIGHTS["audio"] +
        visual_score * WEIGHTS["visual"] +
        weather_score * WEIGHTS["weather"]
    )

    if total_score < 0.7:
        return "Calm"
    elif total_score < 1.4:
        return "Risky"
    else:
        return "Sensory Overload"


# ================== MAIN LOOP ==================
audio_folder = "audio_samples"  # put your .wav files here
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

# Example local image
image_path = "img.jpg"  # change to your image filename
img_array = load_image_from_file(image_path)
visual_status, visual_reasons, color_tone = classify_visual_features(img_array)

print("\n=== Processing Complete ===")
print("Reports generated:\n")

count = 1
for filename in os.listdir(audio_folder):
    if filename.endswith(".wav"):
        filepath = os.path.join(audio_folder, filename)

        # Process audio
        sound_db = get_sound_db_from_file(filepath)
        audio_status, audio_reasons = classify_audio(sound_db)

        # Process weather
        weather_status, weather_reasons = classify_weather(temp, humidity, uv, pm25)

        # Compute overall environment classification
        overall_status = compute_environment_score(audio_status, visual_status, weather_status)

        # Build report
        report = []
        report.append("========== ENVIRONMENT REPORT ==========")
        report.append(f"File: {filename}")
        report.append("")
        report.append(">> WEATHER INFO")
        report.append(weather_info)
        report.append(f"Status: {weather_status}")
        if weather_reasons:
            report.append("Reasons: " + ", ".join(weather_reasons))
        report.append("")
        report.append(">> AUDIO CLASSIFICATION")
        report.append(f"Sound Level: {sound_db} dB")
        report.append(f"Status: {audio_status}")
        if audio_reasons:
            report.append("Reasons: " + ", ".join(audio_reasons))
        report.append("")
        report.append(">> VISUAL CLASSIFICATION")
        report.append(f"Status: {visual_status}")
        if visual_reasons:
            report.append("Reasons: " + visual_reasons)
        report.append(f"Color Tone: {color_tone}")
        report.append("")
        report.append(">> OVERALL ENVIRONMENT STATUS")
        report.append(f"Classification: {overall_status}")
        report.append("========================================")

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outpath = os.path.join(output_folder, f"report_{count}_{timestamp}.txt")
        with open(outpath, "w", encoding="utf-8") as f:
            f.write("\n".join(report))

        # Clean print (no filenames)
        print(f"✔ Report {count} saved")
        count += 1

