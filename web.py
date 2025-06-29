from face import detect_face_landmarks
from avatar import render_avatar
from fer import FER
import cv2
import numpy as np

# Initialize FER emotion detector
emotion_detector = FER(mtcnn=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect landmarks on the real camera frame
    landmarks, _ = detect_face_landmarks(frame)

    # Detect emotion on the real camera frame
    results = emotion_detector.detect_emotions(frame)
    if results:
        emotions = results[0]['emotions']
        emotion = max(emotions, key=emotions.get)
    else:
        emotion = "neutral"

    # Render avatar based on landmarks and detected emotion
    avatar_frame = render_avatar(landmarks, emotion)

    # Resize both frames to the same height (for alignment)
    desired_height = 480
    frame_width = int(frame.shape[1] * desired_height / frame.shape[0])
    avatar_width = int(avatar_frame.shape[1] * desired_height / avatar_frame.shape[0])

    frame_resized = cv2.resize(frame, (frame_width, desired_height))
    avatar_resized = cv2.resize(avatar_frame, (avatar_width, desired_height))

    # Combine avatar (left) and real camera (right)
    combined_frame = np.hstack((avatar_resized, frame_resized))

    # Display combined frame in one window
    cv2.imshow('Avatar (Left) & Live Camera (Right)', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
