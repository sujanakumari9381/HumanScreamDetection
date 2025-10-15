import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# Paths
DATASET_PATH = "dataset"
POS_PATH = os.path.join(r"C:\Users\Sujana\Documents\HumanScreamDetection\dataSet", "positive")
NEG_PATH = os.path.join(r"C:\Users\Sujana\Documents\HumanScreamDetection\dataSet", "negative")

print("POS_PATH exists:", os.path.exists(POS_PATH))
print("NEG_PATH exists:", os.path.exists(NEG_PATH))

features = []
labels = []

# Extract MFCC features from positive (scream) files
for file in os.listdir(POS_PATH):
    audio_path = os.path.join(POS_PATH, file)
    y, sr = librosa.load(audio_path, duration=3)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    features.append(mfcc)
    labels.append(1)  # scream = 1

# Extract MFCC features from negative (non-scream) files
for file in os.listdir(NEG_PATH):
    audio_path = os.path.join(NEG_PATH, file)
    y, sr = librosa.load(audio_path, duration=3)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    features.append(mfcc)
    labels.append(0)  # non-scream = 0

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple SVM model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Evaluate accuracy
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(model, "scream_model.pkl")
print("Model saved as scream_model.pkl")

