import pyaudio
import numpy as np
import requests

# ================== AUDIO FUNCTION ==================
def get_sound_db(duration=3, rate=44100, chunk=1024):
    """Capture audio from mic and return average dB level"""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("Recording audio...")
    frames = []

    for _ in range(int(rate / chunk * duration)):
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.int16))

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.hstack(frames)
    rms = np.sqrt(np.mean(audio_data**2))     # Root mean square
    db = 20 * np.log10(rms + 1e-6)            # Convert to dB
    return round(db, 2)


# ================== WEATHER API (WeatherAPI.com) ==================
API_KEY = "5b4ce54a13604a8795e105427252108"   # <-- Replace with your WeatherAPI.com key
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

print(f"Weather: {condition}, Temp: {temp}°C, Humidity: {humidity}%, UV: {uv}, PM2.5: {pm25}")

# ================== MAIN CLASSIFICATION ==================
sound_db = get_sound_db(duration=3)
print(f"Sound Level: {sound_db} dB")

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

# If any weather risks found
if reasons and status != "Overload":
    status = "Risky"

print(f"\nFinal Classification -> Environment is: {status}")

if reasons:
    print("Reasons:", ", ".join(reasons))


# ================== VISUAL INPUT (Unsplash API) ==================
UNSPLASH_ACCESS_KEY = "Rxp2tLXDvOFA644pOLS7amxPB02jcxWIRIzdWdqeMUY"   # <-- Replace with Unsplash API Key

def get_image_from_unsplash(query="crowd"):
    """Fetch an image URL from Unsplash API based on query"""
    url = f"https://api.unsplash.com/photos/random?query={query}&client_id={UNSPLASH_ACCESS_KEY}"
    response = requests.get(url).json()
    if "urls" in response:
        return response["urls"]["regular"], query
    return None, query

def classify_visual_overload(query="crowd"):
    """Simple rule-based classification using Unsplash image keywords"""
    image_url, q = get_image_from_unsplash(query)
    if not image_url:
        return "Unknown", "No image fetched", None

    # Rule-based classification
    risky_keywords = ["crowd", "concert", "traffic", "festival", "stadium"]
    overload_keywords = ["strobe", "flash", "fireworks", "billboard"]

    status, reason = "Calm", "Normal scene"

    if q.lower() in risky_keywords:
        status, reason = "Risky", "Crowded / visually stressful"
    if q.lower() in overload_keywords:
        status, reason = "Overload", "Too much visual stimulation"

    return status, reason, image_url


# ==== Example usage ====
status, reason, image_url = classify_visual_overload("concert")
print(f"\nVisual Classification -> {status} ({reason})")
print(f"Image Source -> {image_url}")
