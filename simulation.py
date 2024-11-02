import time
from wisdm_inference_simulator import generate_random_input, run_inference, LABELS
from emotion_detection import continuous_emotion_detection, cleanup
import numpy as np

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
    elif activity == 'Jogging' or activity == 'Walking':
        adjust_temperature(20 if emotion == "happy" else 18)
        set_music("upbeat")
    elif activity == 'Standing':
        adjust_lights("bright")
        adjust_temperature(22)
    elif activity == 'Downstairs' or activity == 'Upstairs':
        adjust_lights("bright" if emotion == "angry" else "soft")
    else:
        print("No specific action for this activity.")
        

if __name__ == 'main':
    previous_prediction = None  # Initialize previous prediction
    while True:
        input_data = generate_random_input()
        prediction = run_inference(input_data)
        predicted_class = np.argmax(prediction)
        emotion = continuous_emotion_detection()
        
        # Check if there is a change from the previous prediction
        if predicted_class != previous_prediction:
            activity = LABELS[predicted_class]
            take_smart_action(activity, emotion)
            previous_prediction = predicted_class
        
        # Wait for 5 seconds before the next check
        time.sleep(3)
        
        cleanup()