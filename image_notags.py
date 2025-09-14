import os
import time
import requests
import numpy as np
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import cv2
import pyaudio

# ------------------ CONFIG ------------------
# Either set WEATHER_API_KEY env var or set below
API_KEY = os.getenv("WEATHER_API_KEY", "5b4ce54a13604a8795e105427252108")
CITY = "Delhi"

# Local fallback image (container path provided by developer)
LOCAL_FALLBACK_IMAGE = "/mnt/data/1763793e-8d7e-45b0-9aed-d55e0f53a033.png"

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
    """Download image and return as RGB numpy array. Raises ValueError on failure."""
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
            # but still try to open if server misreports headers

        # Try Pillow first
        try:
            img = Image.open(BytesIO(resp.content))
            img = img.convert("RGB")
            arr = np.array(img)
            if arr.size == 0:
                raise ValueError("Downloaded image is empty.")
            return arr
        except UnidentifiedImageError:
            # Try OpenCV decode as fallback
            print("[Image] Pillow couldn't identify image, trying OpenCV decode.")
            try:
                arr = np.frombuffer(resp.content, np.uint8)
                cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
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
    """Accepts URL or local path. Returns RGB numpy array or raises ValueError."""
    # Heuristics: if starts with http:// or https:// treat as URL
    if isinstance(path_or_url, str) and path_or_url.lower().startswith(("http://", "https://")):
        return load_image_from_url(path_or_url)
    # else try local path
    if not os.path.exists(path_or_url):
        raise ValueError(f"Local image not found: {path_or_url}")
    # Load local with Pillow then convert
    try:
        img = Image.open(path_or_url).convert("RGB")
        arr = np.array(img)
        if arr.size == 0:
            raise ValueError("Local image is empty.")
        return arr
    except Exception as e:
        # Try cv2 fallback
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
        # fallback using PIL conversion
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

    # Resize if extremely large to speed up processing
    h, w = image_array.shape[:2]
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_array = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

    brightness = get_brightness(image_array)
    saturation = get_saturation(image_array)
    edge_density = get_edge_density(image_array)
    tone = get_color_temperature(image_array)

    status = "Calm"
    reasons = []

    if brightness > 200:
        status = "Overload"
        reasons.append("Too Bright")
    elif brightness < 50:
        reasons.append("Too Dark")

    if saturation > 150:
        if status == "Calm":
            status = "Risky"
        reasons.append("High Saturation / Strong Colors")

    if edge_density > 0.10:
        # >10% edges considered crowded
        status = "Risky" if status == "Calm" else "Overload"
        reasons.append("Crowded Scene (High Edge Density)")

    return status, reasons, tone

# ------------------ MAIN CLASSIFICATION FLOW ------------------
def main():
    # WEATHER
    weather = fetch_weather(API_KEY, CITY)
    if weather is None:
        print("[Main] Weather not available. Proceeding with defaults.")
        temp = humidity = uv = 0
        pm25 = 0
        condition = "Unknown"
    else:
        current = weather.get("current", {})
        temp = current.get("temp_c", 0)
        humidity = current.get("humidity", 0)
        uv = current.get("uv", 0)
        condition = current.get("condition", {}).get("text", "Unknown")
        air_quality = current.get("air_quality", {})
        pm25 = air_quality.get("pm2_5", 0)

    print(f"[Weather] {condition}, Temp: {temp}°C, Humidity: {humidity}%, UV: {uv}, PM2.5: {pm25}")

    # AUDIO
    sound_db = get_sound_db(duration=3)
    if sound_db is None:
        print("[Main] Audio unavailable. Using default calm value.")
        sound_db_val = 30.0
    else:
        sound_db_val = sound_db
    print(f"[Audio] Sound Level: {sound_db_val} dB")

    # BASIC RULES
    status = "Calm"
    reasons = []

    if temp > 32:
        reasons.append("High Temperature")
    if humidity > 70:
        reasons.append("High Humidity")
    if uv > 7:
        reasons.append("Strong UV / Sunlight")
    if pm25 > 50:
        reasons.append("Poor Air Quality")

    if sound_db_val > 85:
        reasons.append("Extreme Noise")
        status = "Overload"
    elif sound_db_val > 70:
        reasons.append("High Noise")
        status = "Risky"

    if reasons and status != "Overload":
        status = "Risky"

    print(f"\nFinal Classification -> Environment is: {status}")
    if reasons:
        print("Reasons:", ", ".join(reasons))

    # VISUAL
    # Example URL (unchanged) - you can change this to your own or a local path
    test_url = "1.jpg"

    try:
        img_array = None
        try:
            img_array = load_image(test_url)
        except Exception as e:
            print(f"[Visual] Failed to load from URL: {e}")
            # Try the local fallback provided by developer
            try:
                print(f"[Visual] Trying local fallback: {LOCAL_FALLBACK_IMAGE}")
                img_array = load_image(LOCAL_FALLBACK_IMAGE)
            except Exception as e2:
                print(f"[Visual] Local fallback failed: {e2}")
                img_array = None

        visual_status, visual_reasons, color_tone = classify_visual_features(img_array) if img_array is not None else ("Unknown", ["No image"], "Neutral")

        print(f"\nVisual Classification -> {visual_status}")
        if visual_reasons:
            print("Reasons:", ", ".join(visual_reasons))
        print(f"Color Tone: {color_tone}")

    except Exception as e:
        print(f"[Visual] Unexpected error: {e}")

if __name__ == "__main__":
    main()
