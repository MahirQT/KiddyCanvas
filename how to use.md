# KiddyCanvas Handwritten Character Recognition

This project implements a Convolutional Neural Network (CNN) for recognizing handwritten characters using the EMNIST dataset. It includes a drawing application that allows users to draw characters and get real-time predictions.

## Requirements

- Python 3.7+
- TensorFlow 2.4.0+
- NumPy
- scikit-learn
- Pillow
- joblib

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the EMNIST Balanced dataset, which should be placed in the `emnist_balanced` directory with the following files:
- emnist-balanced-train-images-idx3-ubyte.gz
- emnist-balanced-train-labels-idx1-ubyte.gz
- emnist-balanced-test-images-idx3-ubyte.gz
- emnist-balanced-test-labels-idx1-ubyte.gz
- emnist-balanced-mapping.txt

## Usage

1. Run the main script:
```bash
python emnist_cnn.py
```

2. The script will:
   - Load and preprocess the EMNIST data
   - Train a CNN model
   - Save the trained model
   - Launch a drawing application

3. In the drawing application:
   - Draw a character using your mouse
   - Click "Predict" to see the model's prediction
   - Click "Clear" to start over

## Model Architecture

The CNN model consists of:
- 2 Convolutional layers with ReLU activation
- 2 MaxPooling layers
- 1 Dense layer with ReLU activation
- Dropout layer for regularization
- Output layer with softmax activation

## Notes

- The model is trained on a subset of 10,000 samples for demonstration purposes
- The drawing canvas is 280x280 pixels, which is resized to 28x28 for prediction
- The model is saved as "cnn_emnist_model.h5" for future use 