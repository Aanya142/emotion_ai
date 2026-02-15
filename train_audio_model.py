import librosa
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

dataset_path = "dataset_audio"

X = []
y = []

for emotion in emotion_labels:
    emotion_path = os.path.join(dataset_path, emotion)

    if not os.path.exists(emotion_path):
        print(f"Skipping {emotion} (folder not found)")
        continue

    for file in os.listdir(emotion_path):
        file_path = os.path.join(emotion_path, file)

        try:
            audio, sr = librosa.load(file_path, duration=3)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc_scaled = np.mean(mfcc.T, axis=0)

            X.append(mfcc_scaled)
            y.append(emotion_labels.index(emotion))

        except Exception as e:
            print("Error loading file:", file_path)

X = np.array(X)
y = to_categorical(y, num_classes=len(emotion_labels))


scaler = StandardScaler()
X = scaler.fit_transform(X)


np.save("audio_scaler_mean.npy", scaler.mean_)
np.save("audio_scaler_scale.npy", scaler.scale_)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)


model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(40,)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(emotion_labels), activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=50,
          batch_size=32,
          validation_data=(X_test, y_test))

model.save("audio_emotion_model.h5")

print("Audio model trained and saved successfully!")
