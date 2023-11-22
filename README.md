# Speech Emotion Recognition using LSTM
This repository contains a deep learning model for Speech Emotion Recognition (SER) using LSTM layers. The model is designed to classify speech recordings into various emotions, including fear, angry, happy, sad, disgust, neutral, and pleasant surprise. The project utilizes Natural Language Processing (NLP) techniques and deep learning to achieve this classification.

## Overview
Speech Emotion Recognition is a vital area in the field of natural language processing and human-computer interaction. This project demonstrates how to build a deep learning model to recognize emotions from speech recordings. The model employs LSTM layers for sequence modeling, allowing it to capture temporal dependencies in the audio data.

## Libraries
To use this project, you will need to have the following libraries installed:

 - librosa

 - seaborn

 - matplotlib

 - scikit-learn

 - pandas

 - IPython

- numpy

 - keras

## Dataset
The dataset used for this project is the TESS(Toronto emotional speech set). It can be found on Kaggle at the following link: [TESS Dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess). It contains speech recordings labeled with emotions. The emotions and their distribution in the dataset are as follows:

 - Fear: 400 samples

 - Angry: 400 samples

 - Happy: 400 samples

 - Sad: 400 samples

 - Disgust: 400 samples

 - Neutral: 400 samples

 - Pleasant Surprise: 400 samples

The dataset is a multiclass classification problem, where the goal is to classify speech recordings into one of these seven emotion categories.

## Model Architecture
The deep learning model used for Speech Emotion Recognition is a sequential model with the following layers:

-LSTM layers for sequence modeling

-Dense layers for classification

-Dropout layers for regularization

This architecture is designed to capture both spatial and temporal features in the audio data.

## Training
The model was trained on the provided dataset, and the training results are as follows:

 - Training accuracy: 98%

 - Test accuracy: 96%

 - Training loss: 0.06

 - Test loss: 0.1

The model is capable of accurately recognizing emotions in speech recordings.
