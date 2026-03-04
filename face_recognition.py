"""
Real-Time Face Recognition App
================================
Uses a pre-trained Keras model (.h5) with OpenCV webcam feed
to detect and recognize faces in real time.

Controls:
  - Press 'q' to quit the application
"""

import os

# Enable legacy Keras for compatibility with older .h5 models
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
import numpy as np
import tensorflow as tf

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

IMG_SIZE = 224                # Model input size (224x224)
CONFIDENCE_THRESHOLD = 0.60  # Minimum confidence to display a prediction

# Colors (BGR format)
BOX_COLOR = (0, 255, 0)       # Green bounding box
TEXT_BG_COLOR = (0, 255, 0)   # Green text background
TEXT_COLOR = (0, 0, 0)        # Black text
LOW_CONF_COLOR = (0, 0, 255)  # Red for low-confidence predictions


# ──────────────────────────────────────────────
# Load Labels
# ──────────────────────────────────────────────
def load_labels(filepath: str) -> list[str]:
    """Parse labels.txt and return a list of class names."""
    labels = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                # Format: "0 ClassName" — split on first space
                parts = line.split(maxsplit=1)
                labels.append(parts[1] if len(parts) > 1 else parts[0])
    return labels


# ──────────────────────────────────────────────
# Preprocess Face for Model
# ──────────────────────────────────────────────
def preprocess_face(face_img: np.ndarray) -> np.ndarray:
    """Resize and normalize a face image for the Keras model."""
    face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_normalized = (face_resized.astype(np.float32) / 127.5) - 1.0  # Normalize to [-1, 1]
    return np.expand_dims(face_normalized, axis=0)  # Add batch dimension


# ──────────────────────────────────────────────
# Draw Prediction on Frame
# ──────────────────────────────────────────────
def draw_prediction(frame, x, y, w, h, label: str, confidence: float):
    """Draw bounding box and label with confidence on the frame."""
    is_confident = confidence >= CONFIDENCE_THRESHOLD
    color = BOX_COLOR if is_confident else LOW_CONF_COLOR

    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Prepare label text
    text = f"{label}: {confidence * 100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    # Get text size for background rectangle
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Draw text background
    cv2.rectangle(
        frame,
        (x, y - text_h - 10),
        (x + text_w + 5, y),
        color,
        cv2.FILLED,
    )

    # Draw text
    cv2.putText(
        frame, text,
        (x + 2, y - 5),
        font, font_scale, TEXT_COLOR, thickness,
    )


# ──────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────
def main():
    # Load model and labels
    print("[INFO] Loading Keras model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    labels = load_labels(LABELS_PATH)
    print(f"[INFO] Loaded {len(labels)} classes: {labels}")

    # Load Haar cascade for face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Start webcam
    print("[INFO] Starting webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check your camera connection.")
        return

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame. Exiting...")
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50),
        )

        # Process each detected face
        for (x, y, w, h) in faces:
            # Crop face region from the original color frame
            face_roi = frame[y : y + h, x : x + w]

            # Preprocess and predict
            processed_face = preprocess_face(face_roi)
            predictions = model.predict(processed_face, verbose=0)
            class_index = np.argmax(predictions[0])
            confidence = predictions[0][class_index]

            # Get label name
            label = labels[class_index] if class_index < len(labels) else "Unknown"

            # Draw on frame
            draw_prediction(frame, x, y, w, h, label, confidence)

        # Display the frame
        cv2.imshow("Face Recognition - Press 'q' to quit", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    print("[INFO] Shutting down...")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
