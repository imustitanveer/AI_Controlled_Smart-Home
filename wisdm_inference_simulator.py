import numpy as np
import time
import tensorflow as tf
import warnings

# Suppress future warnings from numpy
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

# Run the inference simulation if this script is executed as the main module
if __name__ == '__main__':
    previous_prediction = None  # Initialize previous prediction

    # Infinite loop for real-time simulation
    try:
        while True:
            input_data = generate_random_input()
            prediction = run_inference(input_data)
            predicted_class = np.argmax(prediction)
            
            # Check if there is a change from the previous prediction
            if predicted_class != previous_prediction:
                activity = LABELS[predicted_class]
                print(f"Detected change in activity: {activity}")
                previous_prediction = predicted_class
            
            # Wait for 5 seconds before the next check
            time.sleep(3)

    except KeyboardInterrupt:
        print("Inference simulation stopped.")

    finally:
        sess.close()