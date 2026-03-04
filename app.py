"""
Real-Time Face Recognition — Flask Web App
============================================
Streams a live webcam feed with face detection and recognition
to the browser via http://localhost:5000

Controls:
  - Open http://localhost:5000 in your browser
  - Press Ctrl+C in terminal to stop the server
"""

import os

# Enable legacy Keras for compatibility with older .h5 models
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, render_template

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

IMG_SIZE = 224                # Model input size (224x224)
CONFIDENCE_THRESHOLD = 0.60  # Minimum confidence to show prediction

# Colors (BGR)
BOX_COLOR = (0, 255, 0)       # Green
LOW_CONF_COLOR = (0, 0, 255)  # Red
TEXT_COLOR = (0, 0, 0)        # Black


# ──────────────────────────────────────────────
# Load Labels
# ──────────────────────────────────────────────
def load_labels(filepath: str) -> list[str]:
    """Parse labels.txt → list of class names."""
    labels = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(maxsplit=1)
                labels.append(parts[1] if len(parts) > 1 else parts[0])
    return labels


# ──────────────────────────────────────────────
# Preprocess Face
# ──────────────────────────────────────────────
def preprocess_face(face_img: np.ndarray) -> np.ndarray:
    """Resize and normalize a face image for the Keras model."""
    face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_normalized = (face_resized.astype(np.float32) / 127.5) - 1.0
    return np.expand_dims(face_normalized, axis=0)


# ──────────────────────────────────────────────
# Draw Prediction
# ──────────────────────────────────────────────
def draw_prediction(frame, x, y, w, h, label: str, confidence: float):
    """Draw bounding box + label with confidence on the frame."""
    is_confident = confidence >= CONFIDENCE_THRESHOLD
    color = BOX_COLOR if is_confident else LOW_CONF_COLOR

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Label text
    text = f"{label}: {confidence * 100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Text background
    cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 5, y), color, cv2.FILLED)

    # Text
    cv2.putText(frame, text, (x + 2, y - 5), font, font_scale, TEXT_COLOR, thickness)


# ──────────────────────────────────────────────
# Initialize Model & Camera
# ──────────────────────────────────────────────
print("[INFO] Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
labels = load_labels(LABELS_PATH)
print(f"[INFO] Loaded {len(labels)} classes: {labels}")

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

camera = cv2.VideoCapture(0)

# ──────────────────────────────────────────────
# Flask App
# ──────────────────────────────────────────────
app = Flask(__name__)


def generate_frames():
    """Continuously capture frames, detect faces, predict, and yield as MJPEG."""
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

        # Process each face
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            processed = preprocess_face(face_roi)
            predictions = model.predict(processed, verbose=0)
            idx = np.argmax(predictions[0])
            confidence = predictions[0][idx]
            label = labels[idx] if idx < len(labels) else "Unknown"
            draw_prediction(frame, x, y, w, h, label, confidence)

        # Encode frame as JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html", num_classes=len(labels))


@app.route("/video_feed")
def video_feed():
    """Stream the processed video as multipart MJPEG."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ──────────────────────────────────────────────
# Run Server
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Face Recognition Live at:")
    print("  =>  http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
