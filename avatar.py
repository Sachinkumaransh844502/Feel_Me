import cv2
import numpy as np

def render_avatar(landmarks, emotion):
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    if landmarks:
        for x, y in landmarks:
            cx, cy = int(x * 640), int(y * 480)
            cv2.circle(canvas, (cx, cy), 2, get_emotion_color(emotion), -1)

        cv2.putText(canvas, f"Emotion: {emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, get_emotion_color(emotion), 2)

    return canvas

def get_emotion_color(emotion):
    return {
        "happy": (0, 255, 0),        # Green
        "sad": (255, 0, 0),          # Blue
        "surprised": (0, 255, 255),  # Yellow
        "neutral": (200, 200, 200),  # Light gray
        "angry": (0, 0, 255),        # Red
        "confused": (255, 165, 0),   # Orange
        "thinking": (128, 0, 128)    # Purple
    }.get(emotion, (255, 255, 255))  # White fallback
