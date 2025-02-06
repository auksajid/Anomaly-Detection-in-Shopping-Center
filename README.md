# Anomaly-Detection-in-Shopping
Introduction
Human Action Recognition using deep learning models such as Convolutional LSTM (ConvLSTM) and Long-term Recurrent Convolutional Network (LRCN). The goal is to classify short videos into predefined action categories like WalkingWithDog, TaiChi, Swing, and HorseRace.

Explanation

Import Libraries: 

Imports necessary libraries like OpenCV (cv2), TensorFlow (tf), Keras, and others for video processing, model building, and training.

Data Preprocessing:

Frames Extraction: Function frames_extraction reads video files, resizes frames to 64x64, normalizes pixel values, and selects a fixed number of frames (SEQUENCE_LENGTH) to represent the video.
Dataset Creation: Function create_dataset iterates through video files of different action classes, extracts frames using frames_extraction, and organizes them into features (frames), labels (class indices), and video file paths.
Data Splitting: Splits the dataset into training and testing sets using train_test_split from scikit-learn.
Model Building:

ConvLSTM Model: 

Function create_convlstm_model defines the architecture of the ConvLSTM model using Keras layers like ConvLSTM2D, MaxPooling3D, Dropout, Flatten, and Dense. It's designed to capture spatiotemporal features from video sequences.
LRCN Model: Function create_LRCN_model defines the LRCN model architecture using TimeDistributed layers for applying convolutional operations to each frame, followed by LSTM for sequence learning, and a Dense layer for classification.

Model Training:

Compiles both models using categorical_crossentropy loss, Adam optimizer, and accuracy metric.
Trains the models using the training data and EarlyStopping callback to prevent overfitting.

Model Evaluation:

Evaluate the trained models on the testing data using evaluation and print the loss and accuracy.
Saves the models with filenames containing the date, time, loss, and accuracy.

Prediction:

Function predict_single_action takes a video file path and uses the LRCN model to predict the action performed in the video. It preprocesses the video frames similar to the training data and outputs the predicted class name and confidence score.

Implementation Steps

Dataset: Download the UCF50 dataset and place it in the specified directory ("F:/UCF50" in the code).
Environment: Set up a Python environment with the required libraries (TensorFlow, Keras, OpenCV, etc.). Google Colab is recommended as it provides a pre-configured environment.
Code Execution: Execute the code cells in the notebook sequentially. This will extract frames, create the dataset, build and train the models, and evaluate their performance.
Prediction: Use the predict_single_action function to predict the action in a new video by providing its path.

# Human Action Recognition

This project implements human action recognition using ConvLSTM and LRCN models.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- ... (other libraries)

## Installation

1. Install the required libraries:

## Usage

1. Download the UCF50 dataset and place it in the 'F:/UCF50' directory.
2. Open the Jupyter Notebook (Human_Action_Recognition.ipynb).
3. Execute the code cells sequentially to train and evaluate the models.
4. To predict the action in a new video, use the `predict_single_action` function with the video file path.

## Example
python predict_single_action('path/to/your/video.mp4', SEQUENCE_LENGTH)
## Note

- You can modify the `CLASSES_LIST` variable to train the model on different action categories.
- Adjust the hyperparameters (e.g., SEQUENCE_LENGTH, epochs, batch_size) for optimal performance.
