import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import sounddevice as sd
import librosa
import time
import threading

model = load_model("emotion_model.h5")
audio_model = load_model("audio_emotion_model.h5")

audio_preds = np.zeros(7)
audio_running = False


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
prediction_window = deque(maxlen=10)

last_audio_time = 0
audio_preds = np.zeros(7)   


def record_audio_background():
    global audio_preds, audio_running

    audio_running = True
    duration = 3
    fs = 22050

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    audio = recording.flatten()
    features = extract_features(audio, fs)
    audio_preds = audio_model.predict(features)[0]

    audio_running = False


def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return np.reshape(mfcc_scaled, (1, 40))





face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

def draw_label(img, text, x, y):
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    thickness = 1
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    cv2.rectangle(img, (x, y - h - 20), (x + w + 10, y), (20, 20, 20), -1)
    cv2.putText(img, text, (x + 5, y - 5),
                font, font_scale, (0, 255, 180), thickness)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    current_time = time.time()

    if current_time - last_audio_time > 5 and not audio_running:
        thread = threading.Thread(target=record_audio_background)
        thread.start()
        last_audio_time = current_time

    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        preds = model.predict(face)[0]
        prediction_window.append(preds)

        avg_preds = np.mean(prediction_window, axis=0)

       
        combined = (0.7 * avg_preds) + (0.3 * audio_preds)


        final_index = np.argmax(combined)
        emotion = emotion_labels[final_index]
        confidence = combined[final_index] * 100
        

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 180), 2)
        text = f"{emotion} {confidence:.1f}%"
        draw_label(frame, text, x, y)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
