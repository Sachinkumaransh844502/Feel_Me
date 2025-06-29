import mediapipe as mp
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def detect_face_landmarks(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    landmarks = []
    emotion = "neutral"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                landmarks.append((lm.x, lm.y))

        emotion = classify_emotion(landmarks)

    return landmarks, emotion

def classify_emotion(landmarks):
    if len(landmarks) < 468:
        return "neutral"

    # Calculate some key distances/angles (example heuristics)
    mouth_open = landmarks[14][1] - landmarks[13][1]
    left_mouth_corner = landmarks[78][1]
    right_mouth_corner = landmarks[308][1]
    mouth_width = abs(landmarks[308][0] - landmarks[78][0])

    left_eyebrow = landmarks[105][1]
    right_eyebrow = landmarks[334][1]
    left_eye = landmarks[159][1]
    right_eye = landmarks[386][1]

    # Mouth open threshold for surprise
    if mouth_open > 0.025:
        return "surprised"

    # Smile heuristic: mouth corners go up and mouth is wide
    if (left_mouth_corner < landmarks[13][1]) and (right_mouth_corner < landmarks[13][1]) and mouth_width > 0.05:
        return "happy"

    # Frown / sad: eyebrows lowered and mouth corners down
    if (left_eyebrow > left_eye) and (right_eyebrow > right_eye) and (left_mouth_corner > landmarks[13][1]) and (right_mouth_corner > landmarks[13][1]):
        return "sad"

    # Angry: eyebrows lowered and closer together (example)
    brow_distance = abs(landmarks[105][0] - landmarks[334][0])
    if (left_eyebrow > left_eye) and (right_eyebrow > right_eye) and brow_distance < 0.05:
        return "angry"

    # Confused/thinking - one eyebrow raised
    if (left_eyebrow < left_eye and right_eyebrow > right_eye) or (right_eyebrow < right_eye and left_eyebrow > left_eye):
        return "confused"

    return "neutral"
