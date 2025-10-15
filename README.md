
# 🔊 Human Scream Detection using Deep Learning

This project detects **human screams** from audio input using **machine learning and audio feature extraction (MFCC)** techniques.  
It can be used for **crime prevention, emergency monitoring, and safety alert systems**.

---

## 🚀 Features
- Classifies audio as **scream** or **non-scream**
- Uses **MFCC features** for sound analysis
- Trains a **deep learning model** for detection
- Real-time or recorded audio detection supported
- Can be extended for **crime rate analysis**
  
---

## ⚙️ Requirements
Install dependencies using pip:

```bash
pip install numpy pandas librosa tensorflow keras scikit-learn
````

---

## 🧠 Model Training

Run this command to train your model:

```bash
python training.py
```

This will:

* Load audio files from the `dataSet` folder
* Extract MFCC features
* Train a neural network
* Save the model in the project folder

---

## 🔍 Detection / Testing

To test or detect new audio files:

```bash
python detection.py
```

The model will analyze the input sound and classify it as **Scream** or **Non-Scream**.

---

## 🧪 Sample Output

```
Loading model...
Analyzing audio: test_scream.wav
✅ Prediction: Human Scream Detected 🚨
```

---

## 📈 Future Enhancements

* Integrate with live microphone input
* Real-time scream alerts (via email or SMS)
* Extend to **crime rate prediction** based on scream frequency and location data

