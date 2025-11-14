import os
import pandas as pd
from deepface import DeepFace
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from scipy.io import wavfile
import torch

# -----------------------------------------------------------
# 1. AUDIO EMOTION DETECTION (Wav2Vec2 Model)
# -----------------------------------------------------------
def detect_audio_emotion(audio_path):
    try:
        model_name = "superb/wav2vec2-base-superb-er"

        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        sr, wav = wavfile.read(audio_path)
        if wav.ndim > 1:  
            wav = wav[:, 0]  # convert stereo → mono

        inputs = extractor(wav, sampling_rate=sr, return_tensors="pt")
        logits = model(**inputs).logits
        predicted_id = int(torch.argmax(logits))

        emotion_labels = ["angry", "happy", "sad", "neutral", "fear"]

        return emotion_labels[predicted_id]
    except Exception as e:
        return f"Error: {e}"


# -----------------------------------------------------------
# 2. FACIAL EMOTION DETECTION (DeepFace)
# -----------------------------------------------------------
def detect_face_emotion(image_path):
    try:
        result = DeepFace.analyze(img_path=image_path, actions=["emotion"], enforce_detection=False)
        return result["dominant_emotion"]
    except Exception as e:
        return f"Error: {e}"


# -----------------------------------------------------------
# 3. FUSION LOGIC (Weighted Combination)
# -----------------------------------------------------------
def fuse_emotions(face_emotion, audio_emotion):
    if "Error" in face_emotion:
        return audio_emotion
    if "Error" in audio_emotion:
        return face_emotion

    # Priority to face emotion (weight = 0.6)
    # Audio supports confidence (weight = 0.4)
    if face_emotion == audio_emotion:
        return face_emotion
    else:
        return face_emotion  # face emotion dominates


# -----------------------------------------------------------
# 4. PROCESS MULTIPLE INPUTS → SAVE TO CSV
# -----------------------------------------------------------
def process_all(inputs_folder, output_csv):

    data = []
    for file in os.listdir(inputs_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            image_path = os.path.join(inputs_folder, file)

            # audio must have same base name (e.g., sample1.wav → sample1.jpg)
            base = os.path.splitext(file)[0]
            audio_path = os.path.join(inputs_folder, base + ".wav")

            if not os.path.exists(audio_path):
                continue

            face_em = detect_face_emotion(image_path)
            audio_em = detect_audio_emotion(audio_path)
            final_em = fuse_emotions(face_em, audio_em)

            data.append([file, face_em, audio_em, final_em])
            print(f"Processed {file} → Final Emotion: {final_em}")

    df = pd.DataFrame(data, columns=["File", "Face Emotion", "Audio Emotion", "Final Emotion"])
    df.to_csv(output_csv, index=False)

    print("\nCSV Generated:", output_csv)


# -----------------------------------------------------------
# RUN EVERYTHING
# -----------------------------------------------------------
if __name__ == "__main__":
    INPUT_FOLDER = "inputs"
    OUTPUT_CSV = "output/results.csv"

    os.makedirs("output", exist_ok=True)
    process_all(INPUT_FOLDER, OUTPUT_CSV)
