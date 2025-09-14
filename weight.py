import os
import time
import requests
import numpy as np
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import cv2
import pyaudio

# ------------------ CONFIG ------------------
API_KEY = os.getenv("WEATHER_API_KEY", "5b4ce54a13604a8795e105427252108")
CITY = "Delhi"
LOCAL_FALLBACK_IMAGE = "/mnt/data/1763793e-8d7e-45b0-9aed-d55e0f53a033.png"

# Weights reflect importance (tune these)
FACTOR_WEIGHTS = {
    "temperature": 2.0,
    "humidity": 1.5,
    "uv": 2.5,
    "air_quality": 3.0,
    "noise": 3.5,
    "brightness": 2.0,
    "saturation": 1.5,
    "edge_density": 2.5,
}

# Thresholds: (risk_threshold, extreme_threshold) for normalization
# If extreme_threshold is None, treat any value > risk_threshold as full (1.0)
THRESHOLDS = {
    "temperature": (32.0, 40.0),    # deg C
    "humidity": (70.0, 90.0),       # %
    "uv": (7.0, 11.0),              # UV Index
    "air_quality": (50.0, 150.0),   # PM2.5 ug/m3
    "noise": (70.0, 90.0),          # dB
    "brightness": (200.0, None),    # pixel intensity (0-255); >200 considered very bright
    "saturation": (150.0, None),    # HSV S
    "edge_density": (0.10, 0.25),   # fraction of pixels that are edges
}

# ------------------ AUDIO FUNCTION ------------------
def get_sound_db(duration=3, rate=44100, chunk=1024):
    """Capture audio from mic and return average dB level. Returns None on failure."""
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)
    except Exception as e:
        print(f"[Audio] Could not open microphone stream: {e}")
        return None

    print("[Audio] Recording...")
    frames = []
    try:
        for _ in range(int(rate / chunk * duration)):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))
    except Exception as e:
        print(f"[Audio] Error while recording: {e}")
        stream.stop_stream()
        stream.close()
        p.terminate()
        return None

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.hstack(frames) if frames else np.array([], dtype=np.int16)
    if audio_data.size == 0:
        print("[Audio] No audio captured.")
        return None

    rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
    db = 20 * np.log10(rms + 1e-6)
    return round(float(db), 2)

# ------------------ WEATHER FETCH ------------------
def fetch_weather(api_key, city="Delhi", timeout=8):
    if not api_key:
        raise ValueError("Weather API key not provided.")
    url = f"http://api.weatherapi.com/v1/current.json"
    params = {"key": api_key, "q": city, "aqi": "yes"}
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if "current" not in data:
            raise ValueError("Unexpected weather response structure.")
        return data
    except Exception as e:
        print(f"[Weather] Error fetching weather data: {e}")
        return None

# ------------------ IMAGE LOADING ------------------
def load_image_from_url(url, timeout=10, max_retries=2):
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
        except Exception as e:
            print(f"[Image] Request error (attempt {attempt}): {e}")
            resp = None

        if resp is None:
            time.sleep(0.5)
            continue

        if resp.status_code != 200:
            print(f"[Image] Status {resp.status_code} for URL (attempt {attempt}).")
            time.sleep(0.5)
            continue

        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type.lower():
            print(f"[Image] Content-Type not image: {content_type} (attempt {attempt}).")

        try:
            img = Image.open(BytesIO(resp.content))
            img = img.convert("RGB")
            arr = np.array(img)
            if arr.size == 0:
                raise ValueError("Downloaded image is empty.")
            return arr
        except UnidentifiedImageError:
            print("[Image] Pillow couldn't identify image, trying OpenCV decode.")
            try:
                arr = np.frombuffer(resp.content, np.uint8)
                cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if cv_img is None:
                    raise ValueError("cv2.imdecode returned None")
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                return cv_img
            except Exception as e:
                print(f"[Image] OpenCV decode failed: {e}")
        except Exception as e:
            print(f"[Image] Pillow open failed: {e}")

        time.sleep(0.5)

    raise ValueError("Failed to load image from URL after retries.")

def load_image(path_or_url):
    if isinstance(path_or_url, str) and path_or_url.lower().startswith(("http://", "https://")):
        return load_image_from_url(path_or_url)
    if not os.path.exists(path_or_url):
        raise ValueError(f"Local image not found: {path_or_url}")
    try:
        img = Image.open(path_or_url).convert("RGB")
        arr = np.array(img)
        if arr.size == 0:
            raise ValueError("Local image is empty.")
        return arr
    except Exception as e:
        bgr = cv2.imread(path_or_url)
        if bgr is None:
            raise ValueError(f"Cannot read local image: {e}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# ------------------ VISUAL FEATURE EXTRACTION ------------------
def get_brightness(image_array):
    if image_array is None or image_array.size == 0:
        return 0.0
    gray = np.mean(image_array, axis=2)
    return float(np.mean(gray))

def get_color_temperature(image_array):
    avg_r = float(np.mean(image_array[:, :, 0]))
    avg_b = float(np.mean(image_array[:, :, 2]))
    if avg_r > avg_b + 20:
        return "Warm"
    elif avg_b > avg_r + 20:
        return "Cool"
    else:
        return "Neutral"

def get_saturation(image_array):
    try:
        hsv = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2HSV)
        return float(np.mean(hsv[:, :, 1]))
    except Exception:
        try:
            pil = Image.fromarray(image_array.astype(np.uint8)).convert("HSV")
            hsv = np.array(pil)
            return float(np.mean(hsv[:, :, 1]))
        except Exception:
            return 0.0

def get_edge_density(image_array):
    try:
        gray = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return float(np.sum(edges > 0) / edges.size)
    except Exception:
        return 0.0

def classify_visual_features(image_array):
    if image_array is None or image_array.size == 0:
        return "Unknown", ["Invalid image"], "Neutral"

    h, w = image_array.shape[:2]
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_array = cv2.resize(image_array, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    brightness = get_brightness(image_array)
    saturation = get_saturation(image_array)
    edge_density = get_edge_density(image_array)
    tone = get_color_temperature(image_array)

    status = "Calm"
    reasons = []

    if brightness > THRESHOLDS["brightness"][0]:
        status = "Overload"
        reasons.append("Too Bright")
    elif brightness < 50:
        reasons.append("Too Dark")

    if saturation > THRESHOLDS["saturation"][0]:
        if status == "Calm":
            status = "Risky"
        reasons.append("High Saturation / Strong Colors")

    if edge_density > THRESHOLDS["edge_density"][0]:
        status = "Risky" if status == "Calm" else "Overload"
        reasons.append("Crowded Scene (High Edge Density)")

    return status, reasons, tone

# ------------------ WEIGHTED SCORING (with HARD OVERRIDE) ------------------
def factor_score(value, factor_name):
    """Return weighted score for a factor using THRESHOLDS and FACTOR_WEIGHTS."""
    weight = FACTOR_WEIGHTS.get(factor_name, 1.0)
    risk_th, extreme_th = THRESHOLDS.get(factor_name, (None, None))

    # If threshold not defined, no contribution
    if risk_th is None:
        return 0.0

    # If value <= risk threshold => no score
    if value <= risk_th:
        return 0.0

    # If no extreme threshold defined, any exceedance maps to full weight
    if extreme_th is None:
        frac = 1.0
    else:
        if value >= extreme_th:
            frac = 1.0
        else:
            # scale linearly between risk_th and extreme_th
            frac = (value - risk_th) / (extreme_th - risk_th)
            frac = max(0.0, min(1.0, frac))

    return weight * frac

def classify_environment_weighted(weather_dict, sound_db, visual_tuple):
    """
    weather_dict: {"temp":..., "humidity":..., "uv":..., "pm25":...}
    visual_tuple: (brightness, saturation, edge_density, tone)
    Returns: status, contributions_list, tone, total_score
    """
    contributions = {}

    contributions["temperature"] = factor_score(weather_dict.get("temp", 0.0), "temperature")
    contributions["humidity"] = factor_score(weather_dict.get("humidity", 0.0), "humidity")
    contributions["uv"] = factor_score(weather_dict.get("uv", 0.0), "uv")
    contributions["air_quality"] = factor_score(weather_dict.get("pm25", 0.0), "air_quality")

    contributions["noise"] = factor_score(sound_db if sound_db is not None else 0.0, "noise")

    brightness, saturation, edge_density, tone = visual_tuple
    contributions["brightness"] = factor_score(brightness, "brightness")
    contributions["saturation"] = factor_score(saturation, "saturation")
    contributions["edge_density"] = factor_score(edge_density, "edge_density")

    total_score = sum(contributions.values())

    # --- NEW: Hard rule override ---
    overload_flag = False
    # check actual raw values against extreme thresholds
    raw_values = {
        "temperature": weather_dict.get("temp", 0.0),
        "humidity": weather_dict.get("humidity", 0.0),
        "uv": weather_dict.get("uv", 0.0),
        "air_quality": weather_dict.get("pm25", 0.0),
        "noise": sound_db if sound_db is not None else 0.0,
        "brightness": brightness,
        "saturation": saturation,
        "edge_density": edge_density,
    }
    for factor, (risk_th, extreme_th) in THRESHOLDS.items():
        if extreme_th is None:
            # no extreme threshold defined -> skip hard-override for this factor
            continue
        val = raw_values.get(factor, 0.0)
        if val >= extreme_th:
            overload_flag = True
            break

    # Classification thresholds for aggregated score (tune as needed)
    if overload_flag:
        status = "Overload"
    elif total_score >= 10.0:
        status = "Overload"
    elif total_score >= 4.0:
        status = "Risky"
    else:
        status = "Calm"

    # Format contributions as readable strings
    contrib_list = [(k, float(v)) for k, v in contributions.items() if v > 0]
    contrib_list.sort(key=lambda x: x[1], reverse=True)

    return status, contrib_list, tone, total_score

# ------------------ MAIN FLOW ------------------
def main():
    # WEATHER
    weather = fetch_weather(API_KEY, CITY)
    if weather is None:
        print("[Main] Weather not available. Proceeding with defaults.")
        temp = humidity = uv = 0.0
        pm25 = 0.0
        condition = "Unknown"
    else:
        current = weather.get("current", {})
        temp = current.get("temp_c", 0.0)
        humidity = current.get("humidity", 0.0)
        uv = current.get("uv", 0.0)
        condition = current.get("condition", {}).get("text", "Unknown")
        air_quality = current.get("air_quality", {})
        pm25 = air_quality.get("pm2_5", 0.0)

    print(f"[Weather] {condition}, Temp: {temp}°C, Humidity: {humidity}%, UV: {uv}, PM2.5: {pm25}")

    # AUDIO
    sound_db = get_sound_db(duration=3)
    if sound_db is None:
        print("[Main] Audio unavailable. Using default calm value.")
        sound_db_val = 30.0
    else:
        sound_db_val = sound_db
    print(f"[Audio] Sound Level: {sound_db_val} dB")

    # QUICK RULE CHECK
    reasons = []
    if temp > THRESHOLDS["temperature"][0]:
        reasons.append("High Temperature")
    if humidity > THRESHOLDS["humidity"][0]:
        reasons.append("High Humidity")
    if uv > THRESHOLDS["uv"][0]:
        reasons.append("Strong UV / Sunlight")
    if pm25 > THRESHOLDS["air_quality"][0]:
        reasons.append("Poor Air Quality")
    if sound_db_val > THRESHOLDS["noise"][0]:
        reasons.append("Loud Noise")

    if reasons:
        print(f"(Quick rule-check) Reasons: {', '.join(reasons)}")

    # VISUAL
    test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/640px-Fronalpstock_big.jpg"
    img_array = None
    try:
        try:
            img_array = load_image(test_url)
        except Exception as e:
            print(f"[Visual] Failed to load from URL: {e}")
            # try local fallback
            try:
                print(f"[Visual] Trying local fallback: {LOCAL_FALLBACK_IMAGE}")
                img_array = load_image(LOCAL_FALLBACK_IMAGE)
            except Exception as e2:
                print(f"[Visual] Local fallback failed: {e2}")
                img_array = None
    except Exception as e:
        print(f"[Visual] Unexpected image loading error: {e}")
        img_array = None

    if img_array is None:
        visual_status, visual_reasons, color_tone = "Unknown", ["No image"], "Neutral"
        brightness = saturation = edge_density = 0.0
    else:
        visual_status, visual_reasons, color_tone = classify_visual_features(img_array)
        brightness = get_brightness(img_array)
        saturation = get_saturation(img_array)
        edge_density = get_edge_density(img_array)

    print(f"\nVisual quick classification -> {visual_status}")
    if visual_reasons:
        print("Visual reasons:", ", ".join(visual_reasons))
    print(f"Color Tone: {color_tone}")

    # WEIGHTED CLASSIFICATION
    weather_dict = {"temp": temp, "humidity": humidity, "uv": uv, "pm25": pm25}
    visual_tuple = (brightness, saturation, edge_density, color_tone)
    final_status, contributions, tone, total_score = classify_environment_weighted(weather_dict, sound_db_val, visual_tuple)

    print(f"\nWeighted Classification -> {final_status} (score = {total_score:.2f})")
    if contributions:
        print("Per-factor contributions (highest -> lowest):")
        for k, v in contributions:
            print(f"  - {k}: {v:.2f} (weight={FACTOR_WEIGHTS.get(k, 'n/a')})")
    else:
        print("No factor contributed to overload score.")
    print(f"Final color tone: {tone}")

if __name__ == "__main__":
    main()
