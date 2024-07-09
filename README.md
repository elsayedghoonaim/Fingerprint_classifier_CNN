# Fingerprint Classification Project

This project implements a deep learning model to classify fingerprints based on gender and finger position using a convolutional neural network (CNN).

## Overview

The model is trained on the SOCOFing dataset to perform two tasks:
1. Binary classification of gender (male/female)
2. Multi-class classification of finger position (10 classes: 5 fingers for each hand)

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- OpenCV (cv2)
- Matplotlib
- Scikit-learn

## Dataset

The project uses the SOCOFing dataset, which includes:
- Real fingerprint images
- Altered fingerprint images (medium alteration)

## Model Architecture

The CNN architecture consists of:
- Convolutional layers
- MaxPooling layers
- Dropout for regularization
- Separate output layers for gender and finger classification

## Features

- Data loading and preprocessing
- Model training with early stopping based on finger classification accuracy
- Evaluation on test set
- Visualization of training history

## Usage

1. Ensure all required libraries are installed.
2. Update the `data_dirs` variable with the correct paths to your SOCOFing dataset.
3. Run the script to train and evaluate the model.

## Results

The model achieves:
- 100% accuracy on gender classification
- Approximately 89% accuracy on finger position classification

## Visualization

The script includes visualizations for:
- Sample fingerprint images from the dataset
- Training and validation accuracy/loss curves

## Note

This project uses mixed precision training to optimize performance on compatible GPUs.

## Future Improvements

- Fine-tuning hyperparameters
- Experimenting with different model architectures
- Implementing data augmentation techniques

## License

[Include your chosen license here]
