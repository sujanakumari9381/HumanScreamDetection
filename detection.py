import os
import librosa
import numpy as np
import joblib

# Load trained model
model = joblib.load("scream_model.pkl")

def detect_scream(audio_path):
    # Check if file exists
    if not os.path.isfile(audio_path):
        print(f"❌ File not found: {audio_path}")
        return

    try:
        # Load audio (first 3 seconds)
        y, sr = librosa.load(audio_path, duration=3)
        # Extract MFCC features
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        mfcc = mfcc.reshape(1, -1)
        # Make prediction
        prediction = model.predict(mfcc)

        if prediction[0] == 1:
            print("Scream detected!")
        else:
            print("Normal sound (no scream).")
    except Exception as e:
        print(f"❌ Error processing file: {e}")

audio_file = r"C:\Users\Sujana\Documents\HumanScreamDetection\dataSet\dataSet\positive\9sbiN76Znck_out.wav"
detect_scream(audio_file)
