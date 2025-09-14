import numpy as np
import requests
from PIL import Image
from io import BytesIO
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


# ================== AUDIO FUNCTION ==================
def get_sound_db_from_file(filepath):
    """Read WAV file and return average dB level"""
    rate, data = wavfile.read(filepath)

    # Stereo -> mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    rms = np.sqrt(np.mean(data**2))
    db = 20 * np.log10(rms + 1e-6)
    return round(db, 2)


def classify_audio(sound_db, temp, humidity, uv, pm25):
    status = "Calm"
    reasons = []

    # Weather stress rules
    if temp > 32:
        reasons.append("High Temperature")
    if humidity > 70:
        reasons.append("High Humidity")
    if uv > 7:
        reasons.append("Strong UV / Sunlight")
    if pm25 > 50:
        reasons.append("Poor Air Quality")

    # Noise thresholds
    if sound_db > 85:
        reasons.append("Extreme Noise")
        status = "Overload"
    elif sound_db > 70:
        reasons.append("High Noise")
        status = "Risky"

    if reasons and status != "Overload":
        status = "Risky"

    return status, reasons


# ================== VISUAL INPUT ==================
def load_image_from_url(url):
    """Download image and return as numpy array"""
    resp = requests.get(url)
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return np.array(img)


def get_brightness(image_array):
    gray = np.mean(image_array, axis=2)
    return np.mean(gray)


def get_color_temperature(image_array):
    avg_r = np.mean(image_array[:, :, 0])
    avg_b = np.mean(image_array[:, :, 2])
    if avg_r > avg_b + 20:
        return "Warm"
    elif avg_b > avg_r + 20:
        return "Cool"
    else:
        return "Neutral"


def get_saturation(image_array):
    hsv = np.array(Image.fromarray(image_array).convert("HSV"))
    return np.mean(hsv[:, :, 1])


def get_edge_density(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size


def classify_visual_features(image_array):
    brightness = get_brightness(image_array)
    saturation = get_saturation(image_array)
    edge_density = get_edge_density(image_array)
    temp = get_color_temperature(image_array)

    status, reasons = "Calm", []

    if brightness > 200:
        status, reasons = "Overload", ["Too Bright"]
    elif brightness < 50:
        reasons.append("Too Dark")

    if saturation > 150:
        if status == "Calm":
            status = "Risky"
        reasons.append("High Saturation / Strong Colors")

    if edge_density > 0.1:
        status = "Risky" if status == "Calm" else "Overload"
        reasons.append("Crowded Scene (High Edge Density)")

    return status, reasons, temp


# ================== MAIN LOOP ==================
audio_folder = "audio_samples"  # put your .wav files here
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

# Example image
test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/640px-Fronalpstock_big.jpg"
img_array = load_image_from_url(test_url)
visual_status, visual_reasons, color_tone = classify_visual_features(img_array)

for filename in os.listdir(audio_folder):
    if filename.endswith(".wav"):
        filepath = os.path.join(audio_folder, filename)

        # Process audio
        sound_db = get_sound_db_from_file(filepath)
        audio_status, audio_reasons = classify_audio(sound_db, temp, humidity, uv, pm25)

        # Build report
        report = []
        report.append("========== ENVIRONMENT REPORT ==========")
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
        if visual_reasons:
            report.append("Reasons: " + ", ".join(visual_reasons))
        report.append(f"Color Tone: {color_tone}")
        report.append("========================================")

        # Save to text file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outpath = os.path.join(output_folder, f"{filename}_{timestamp}.txt")
        with open(outpath, "w") as f:
            f.write("\n".join(report))

        print(f"Processed {filename} -> saved report at {outpath}")