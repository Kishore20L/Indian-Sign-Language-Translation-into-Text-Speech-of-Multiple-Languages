import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import threading
from gtts import gTTS
from playsound import playsound
from PIL import Image, ImageDraw, ImageFont

# Load the model
def load_model(model_file_path):
    if os.path.exists(model_file_path):
        try:
            model_dict = pickle.load(open(model_file_path, 'rb'))
            model = model_dict['model']
            print("✅ Model loaded successfully!")
            return model
        except Exception as e:
            print("❌ Error loading model:", e)
    else:
        print("❌ Model file does not exist:", model_file_path)
    return None

# Specify the model path
model_file_path = 'C:/Users/Dell/Downloads/sign-language-detector-python-master/sign-language-detector-python-master/model.p'
model = load_model(model_file_path)
if model is None:
    exit()

# Initialize webcam properly
def initialize_webcam():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not access webcam. Trying alternative camera index...")
        cap = cv2.VideoCapture(1)  # Try another camera index

    if not cap.isOpened():
        print("❌ Error: No available webcam detected. Exiting...")
        exit()

    print("✅ Webcam initialized successfully!")
    return cap

# Load webcam
cap = initialize_webcam()

# Supported languages
languages = ["te", "hi", "ta", "en"]  # Telugu, Hindi, Tamil, English
language_index = 0  # Start with Telugu

# Labels for different languages
labels_dict = {
    "te": {0: "హలో", 1: "నేను", 2: "పురుషుడు"},
    "hi": {0: "नमस्ते", 1: "मैं", 2: "पुरुष"},
    "ta": {0: "வணக்கம்", 1: "நான்", 2: "ஆண்"},
    "en": {0: "Hello", 1: "I", 2: "Man"}
}

detected_signs = []

# Font Paths (Modify these to match your system)
font_paths = {
    "te": "C:/Users/Dell/Downloads/fonts/Mallanna.ttf",
    "hi": "C:/Users/Dell/Downloads/Noto_Sans_Devanagari.tff",
    "ta": "C:/Users/Dell/Downloads/Noto_Sans_Tamil (1).tff",
    "en": "C:/Windows/Fonts/Arial.ttf"
}

# Function to load fonts dynamically
def get_font(lang, size=32):
    return ImageFont.truetype(font_paths[lang], size)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=2)

# Function to play detected sign as audio
def play_audio(text, lang):
    def run():
        try:
            tts = gTTS(text=text, lang=lang)
            audio_file = "temp_audio.mp3"
            tts.save(audio_file)
            playsound(audio_file)
            os.remove(audio_file)
        except Exception as e:
            print("❌ Error in audio playback:", e)
    
    thread = threading.Thread(target=run)
    thread.start()

# Function to display text on frame
def put_text(img, text, position, lang):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_font(lang, size=32)
    draw.text(position, text, font=font, fill=(0, 0, 0, 255))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Function to change language
def change_language():
    global language_index
    language_index = (language_index + 1) % len(languages)
    print(f"🌍 Language changed to {languages[language_index]}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read frame from webcam.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

            x_, y_, data_aux = [], [], []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])
            lang_code = languages[language_index]
            predicted_character = labels_dict[lang_code].get(int(prediction[0]), 'Unknown')

            if predicted_character not in detected_signs:
                detected_signs.append(predicted_character)
                play_audio(predicted_character, lang_code)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            frame = put_text(frame, predicted_character, (x1, y1 - 10), lang_code)

    # Draw Refresh Button
    cv2.rectangle(frame, (10, 10), (100, 50), (255, 255, 255), -1)
    cv2.putText(frame, "Refresh", (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Display concatenated detected signs
    concatenated_signs = ' '.join(detected_signs)
    frame = put_text(frame, concatenated_signs, (10, 80), languages[language_index])

    # Display current language
    cv2.putText(frame, f"Language: {languages[language_index]}", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Sign Language Detector', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('l'):  # Press 'L' to change language dynamically
        change_language()

cap.release()
cv2.destroyAllWindows()
