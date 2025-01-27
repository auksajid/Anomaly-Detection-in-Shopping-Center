# Anomaly-Detection-in-Shopping
A New Approach of Anomaly Detection in Shopping Center Surveillance Videos for Theft Prevention based on RLCNN Model
1. Data Preparation:

The code starts by importing necessary libraries like OpenCV (cv2), TensorFlow (tf), and others for video processing, machine learning, and visualization.
It sets a seed for reproducibility of results.
frames_extraction function: This function takes a video path as input. It reads the video, resizes each frame to 64x64 pixels, normalizes the pixel values to be between 0 and 1, and then appends these preprocessed frames to a list. It returns the list of frames.
create_dataset function: This function iterates through the specified classes (e.g., "WalkingWithDog", "TaiChi"). It finds all videos belonging to each class, extracts frames using the frames_extraction function, and stores them along with their corresponding labels (class index) and video paths.
The create_dataset function is called to create the dataset.
The labels are converted into one-hot encoded vectors using to_categorical.
The dataset is split into training and testing sets using train_test_split.
2. Model Building and Training:

create_convlstm_model and create_LRCN_model functions define the architectures of two different models: ConvLSTM and LRCN (Long-term Recurrent Convolutional Network). Both models are designed for video action recognition. The architectures include layers like ConvLSTM2D, MaxPooling3D, TimeDistributed, Conv2D, MaxPooling2D, LSTM, Flatten, and Dense.
The models are created using the respective functions.
The models are compiled using the Adam optimizer, categorical cross-entropy loss function, and accuracy as the evaluation metric.
The models are trained using the fit method, with early stopping to prevent overfitting.
The trained models are saved to files.
3. Model Evaluation and Prediction:

The trained models are evaluated on the test set using the evaluate method.
Functions like plot_metric visualize the training and validation metrics (loss and accuracy).
predict_single_action function: This function takes a video file path and sequence length as input. It extracts frames from the video, preprocesses them (resizing and normalization), feeds them to the LRCN model for prediction, and prints the predicted action and its confidence score.
Finally, the predict_single_action function performs action recognition on a test video.
In essence, the code performs the following steps:

Loads and preprocesses video data for action recognition.
Builds and trains two deep learning models (ConvLSTM and LRCN).
Evaluates the performance of the trained models.
Uses the LRCN model to predict actions in new videos.
