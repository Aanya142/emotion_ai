import os
import shutil

source_path = "RAVDESS"   # or RAVDESS if renamed
target_path = "dataset_audio"

emotion_map = {
    "01": "Neutral",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprise"
}

for actor_folder in os.listdir(source_path):
    actor_path = os.path.join(source_path, actor_folder)

    # ✅ Skip non-directories (.DS_Store etc.)
    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if not file.endswith(".wav"):
            continue

        parts = file.split("-")
        emotion_code = parts[2]

        if emotion_code in emotion_map:
            emotion = emotion_map[emotion_code]
            src_file = os.path.join(actor_path, file)
            dst_file = os.path.join(target_path, emotion, file)

            shutil.copy(src_file, dst_file)

print("Files organized successfully!")
