import streamlit as st
import tensorflow as tf
import time
import numpy as np
import cv2
from deepface import DeepFace
import warnings

tf.config.run_functions_eagerly(True)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define the labels
LABELS = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']

# Enable TensorFlow 1.x compatibility mode
tf.compat.v1.disable_eager_execution()

# Reset the default graph
tf.compat.v1.reset_default_graph()

# Load the TensorFlow 1.x model in compatibility mode
sess = tf.compat.v1.Session()
saver = tf.compat.v1.train.import_meta_graph('model/wisdm_lstm.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('model/'))

# Initialize all variables (important step to avoid FailedPreconditionError)
sess.run(tf.compat.v1.global_variables_initializer())

# Get the default graph and retrieve input/output tensors
graph = tf.compat.v1.get_default_graph()
X = graph.get_tensor_by_name("input:0")  # Input placeholder
pred_softmax = graph.get_tensor_by_name("y_:0")  # Output placeholder

# Function to generate random input data
def generate_random_input():
    return np.random.rand(1, 90, 3)

# Define a function to make predictions
def run_inference(input_data):
    prediction = sess.run(pred_softmax, feed_dict={X: input_data})
    return prediction

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

# Simulated smart home actions
def adjust_lights(brightness):
    print(f"Adjusting lights to {brightness} brightness.")

def adjust_temperature(setting):
    print(f"Setting temperature to {setting} degrees.")

def set_music(mood):
    print(f"Playing {mood}-themed music.")

# Decision-making function
def take_smart_action(activity, emotion):
    if activity == 'Sitting':
        adjust_lights("dim") if emotion == "sad" else adjust_lights("medium")
        set_music("relaxing") if emotion == "neutral" else set_music("cheerful")
        placeholder1 = st.empty()
        placeholder1 = st.image('images/image2.jpg')
    elif activity == 'Jogging' or activity == 'Walking':
        adjust_temperature(20 if emotion == "happy" else 18)
        set_music("upbeat")
        placeholder1 = st.empty()
        placeholder1 = st.image('images/image4.jpg')
    elif activity == 'Standing':
        adjust_lights("bright")
        adjust_temperature(22)
        placeholder1 = st.empty()
        placeholder1 = st.image('images/image3.jpg')
    elif activity == 'Downstairs' or activity == 'Upstairs':
        adjust_lights("bright" if emotion == "angry" else "soft")
    else:
        print("No specific action for this activity.")
        

# Streamlit app setup
st.title("Smart Home Simulation with Activity and Emotion Detection")

placeholder1 = st.empty()
placeholder1 = st.image('images/image1.jpg')

# Button to start the simulation
if st.button("Run Simulation"):
    previous_prediction = None  # Initialize previous prediction
    st.write("Starting smart home simulation...")

    while True:
        # Run WISDM model inference
        input_data = generate_random_input()
        prediction = run_inference(input_data)
        predicted_class = np.argmax(prediction)
        activity = LABELS[predicted_class]

        # Run emotion detection
        emotion = detect_emotions()

        # Take action if activity or emotion changes
        if predicted_class != previous_prediction:
            if emotion:
                st.write(f"Detected Activity: {activity}")
                st.write(f"Detected Emotion: {emotion}")
                take_smart_action(activity, emotion)
                previous_prediction = predicted_class

        # Delay for demonstration purposes
        time.sleep(3)