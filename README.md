<div align='center'>
 
# A Deep Learning-Based Speech Emotion Recognition Framework
This repository contains a deep learning model for Speech Emotion Recognition (SER) using LSTM layers. The model is designed to classify speech recordings into various emotions, including fear, angry, happy, sad, disgust, neutral, and pleasant surprise. The project utilizes Natural Language Processing (NLP) techniques and deep learning to achieve this classification.
</div>

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

 - LSTM layers for sequence modeling

 - Dense layers for classification

 - Dropout layers for regularization

This architecture is designed to capture both spatial and temporal features in the audio data.

## Training
The model was trained on the provided dataset, and the training results are as follows:
### Accuracy
 - Training accuracy: 98%

 - Test accuracy: 96%

 ![Accuracy](https://github.com/Sukanyasingh3/Speech-Emotion-Recognition/assets/113462236/848a44a0-80d6-4d93-92fe-5459805e7be9)
 
### Loss
 - Training loss: 0.06

 - Test loss: 0.1

![Loss](https://github.com/Sukanyasingh3/Speech-Emotion-Recognition/assets/113462236/ad7c6fd3-1160-4781-adf0-59ffe020e8ca)

The model is capable of accurately recognizing emotions in speech recordings.

# Contributing

![gif3](https://github.com/Sukanyasingh3/Speech-Emotion-Recognition/assets/113462236/4599d2da-9ef7-4ad9-8614-5d80817c5f57)


If you would like to contribute to the project, follow these steps:

 - Fork the repository.
 - Create a new branch for your feature or bug fix.
 - Make your changes and submit a pull request.
