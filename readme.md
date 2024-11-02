# AI-Controlled Smart Home

Welcome to the **AI-Controlled Smart Home** project! This smart home system uses artificial intelligence to enhance user experience by adapting the environment based on activity patterns and emotions. The system is trained on the WISDM dataset to predict user activity, employs deep learning for emotion recognition, and adjusts temperature, lighting, and music accordingly. A Streamlit web application is included for simulating real-world scenarios of the smart home.

## Features

1. **Activity Prediction**: An LSTM model trained on the WISDM dataset is used to predict user activities, such as walking, jogging, sitting, etc.
2. **Emotion Detection**: Leveraging DeepFace, the system detects the user's emotional state, adapting the home environment accordingly.
3. **Environment Control**:
   - **Temperature**: Adjusts the temperature based on the user’s detected emotions and predicted activity level.
   - **Lighting**: Modifies lighting intensity and color to create a relaxing or energizing atmosphere.
   - **Music**: Selects and plays music that complements the user’s mood and activity.

4. **Streamlit Web Application**: A user-friendly simulation that shows how the AI-Controlled Smart Home would operate in a real-world environment.

## Technology Stack

- **Python**: Core programming language.
- **LSTM Model**: Trained on the WISDM dataset to predict user activities.
- **DeepFace**: Detects user emotions from facial expressions.
- **Streamlit**: Web application framework for simulating smart home responses.
- **TensorFlow/Keras**: For deep learning model implementation.
- **OpenCV**: Used in combination with DeepFace for image processing.

## Getting Started

### Prerequisites

1. **Python 3.8+**: Ensure Python is installed.
2. **Libraries**:
   ```bash
   pip install tensorflow deepface streamlit opencv-python
   ```
3. **WISDM Dataset**: Download the [WISDM dataset](https://www.cis.fordham.edu/wisdm/dataset.php) and place it in the `/data` directory for training the activity prediction model.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/AI_Controlled_Smart-Home.git
   cd AI_Controlled_Smart-Home
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training the LSTM Model

1. Prepare the WISDM dataset in the required format.
2. Run the training script:
   ```bash
   python train_lstm.py
   ```
   This will train the LSTM model and save it to the `/models` directory.

### Running the Simulation

1. Start the Streamlit web app:
   ```bash
   streamlit run app.py
   ```
2. Use the app to simulate real-world scenarios where the AI system adjusts settings based on detected emotions and activities.

## Project Structure

```plaintext
├── data                    # WISDM dataset and preprocessed data
├── models                  # Saved models
├── app.py                  # Streamlit web app
├── train_lstm.py           # Script to train LSTM model on WISDM data
├── README.md               # Project documentation
└── requirements.txt        # Project dependencies
```

## Future Improvements

- **Voice Control Integration**: Add voice control for manual adjustments.
- **Additional Sensors**: Integrate additional sensors for environmental factors (e.g., CO2 levels).
- **Improved Personalization**: Enhance the model to learn user preferences over time.

## Acknowledgments

- **WISDM Lab** for the dataset.
- **DeepFace** for emotion detection capabilities.

Enjoy your AI-Controlled Smart Home experience!
