# Anomaly-Detection-in-Shopping
A New Approach of Anomaly Detection in Shopping Center Surveillance Videos for Theft Prevention based on RLCNN Model
# Action Recognition using Deep Learning

This repository contains code for action recognition using deep learning models such as ConvLSTM and LRCN. The code is implemented in Python using TensorFlow and Keras.

Dataset

The code uses the UCF50 dataset for action recognition. You can download the dataset from the following link:

[UCF50 Dataset](https://www.crcv.ucf.edu/data/UCF50.php)

Requirements

To run the code, you need to have the following libraries installed:

- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib
- MoviePy
- Scikit-learn

You can install these libraries using pip:

Usage

1. Data Preparation:**
   - Download the UCF50 dataset and extract it to a directory.
   - Update the `DATASET_DIR` variable in the code to point to the directory containing the dataset.
   - Update the `CLASSES_LIST` variable to specify the classes you want to use for training.

2. Model Training:**
   - Run the code to train the ConvLSTM or LRCN model.
   - The trained model will be saved to a file.

3. Action Prediction:**
   - Update the `input_video_file_path` variable in the code to specify the path of the video you want to predict the action for.
   - Run the code to perform action prediction.
   - The predicted action and confidence will be displayed.

Customization

You can customize the code by changing the following parameters:

- `IMAGE_HEIGHT`: Height of the video frames.
- `IMAGE_WIDTH`: Width of the video frames.
- `SEQUENCE_LENGTH`: Number of frames to be fed to the model as one sequence.
- `DATASET_DIR`: Directory containing the UCF50 dataset.
- `CLASSES_LIST`: List of classes used for training.

## Results

The code achieves high accuracy on the UCF50 dataset for action recognition.

