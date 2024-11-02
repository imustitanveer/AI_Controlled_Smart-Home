import cv2
from deepface import DeepFace
import warnings
import time

warnings.simplefilter(action='ignore', category=FutureWarning)

# Initialize face cascade classifier and video capture only once
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

def detect_emotions():
    # Capture a single frame
    ret, frame = cap.read()
    if not ret:
        return None  # If frame capture fails, return None

    # Convert frame to grayscale and then to RGB format
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process only the first detected face for efficiency
    for (x, y, w, h) in faces:
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion  # Return the detected emotion

    return None  # Return None if no face is detected

# Cleanup function to release resources
def cleanup():
    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == '__main__':
    try:
        while True:
            emotion = detect_emotions()
            if emotion:
                print(f"Detected emotion: {emotion}")
            else:
                print("No face detected.")

            # Small delay for demonstration purposes
            time.sleep(2)
    except KeyboardInterrupt:
        print("Emotion detection stopped.")
    finally:
        cleanup()
