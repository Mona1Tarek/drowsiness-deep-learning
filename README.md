# Drowsiness Detection

## Overview
This project implements a **Drowsiness Detection System** that monitors a driver's eyes to detect signs of drowsiness in real time. It leverages **computer vision and deep learning** to analyze eye states (open/closed) and provide alerts when drowsiness is detected.

## Features
- **Real-time eye state detection** using OpenCV and TensorFlow.
- **Drowsiness alert system** with sound notifications.
- **Works with live webcam feed**.
- **Lightweight and efficient model using TFLite for quick inference**.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Mona1Tarek/drowsiness-deep-learning.git
   cd drowsiness_deeplearning
   ```
2. Install required dependencies:
   ```bash
   pip install tensorflow opencv-python numpy playsound
   ```

## Files in the Repository
- `alarm2.mp3`: Alarm sound file for drowsiness alert.
- `detection2.py`: Main script for real-time drowsiness detection.
- `mainMediapipe_samples.py`: Script using Mediapipe for face and eye tracking.
- `optimized_eye_detection_model.tflite`: Optimized TensorFlow Lite model for eye detection.

## Usage
### Run Drowsiness Detection
```bash
python detection.py
```

1. The script will access the webcam and monitor the user's eyes.
2. If the system detects prolonged eye closure, an **alert sound** will be triggered.
3. Press **'q'** to exit the program.

