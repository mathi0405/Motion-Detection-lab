# Motion Recognition using IMU Sensor Fusion

This project involves building a real-time motion classification system using a Raspberry Pi and the Sense HAT. It uses accelerometer and gyroscope data to detect four motion types and displays the results through the LED matrix.

## Overview

The system recognizes the following motion gestures:
- move_none
- move_circle
- move_shake
- move_twist

A neural network model is trained on 1-second sequences (50 time steps, 6 features: accel + gyro) and then converted to TensorFlow Lite for deployment on Raspberry Pi.

## Project Files

- `capture.py`: Script for collecting labeled IMU data (accelerometer + gyroscope).
- `predict.py`: Script for running the TFLite model on Raspberry Pi and controlling the LED matrix based on the prediction.
- `motion_model.tflite`: Trained TensorFlow Lite model used for inference.
- `Lab05_Gesture_Train.ipynb`: Jupyter notebook for training the neural network and converting it to TFLite.
- `motion_data/`: Folder for storing collected IMU samples (organized by label).
- `images/`: Contains pictures showing the LED matrix with motion class predictions.
- `requirements.txt`: Lists required Python libraries.
- `.gitignore`: Specifies which files and folders to ignore in version control.

## Setup Instructions

1. Install Python packages listed in `requirements.txt`.
2. Run `capture.py` on Raspberry Pi to collect labeled motion data.
3. Train the model using `Lab05_Gesture_Train.ipynb` in Google Colab or locally.
4. Convert the model to `.tflite` format.
5. Run `predict.py` to start real-time inference and LED feedback.

## Hardware Used

- Raspberry Pi (with Sense HAT)
- Built-in IMU sensors: Accelerometer and Gyroscope
- LED matrix display

## Motion-to-Color Mapping

- move_none → Black (Off)
- move_circle → Red
- move_shake → Green
- move_twist → Blue

## Image Previews

Images of the LED matrix display during real-time gesture prediction are included in the `images/` folder:
- move_circle (Red LED)
- move_shake (Green LED)
- move_twist (Blue LED)
- Idle or no motion (White or Off)
