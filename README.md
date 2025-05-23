# Deep-Learning Based Pneumonia Diagnosis using chest x-rays
An AI and ML model that uses images from Kaggle to train a model to recognize pneumonia chest X-rays. 

## Project Motivation
AI is being increasingly integrated into medicine, and this was an opportunity to research more about that and apply my knowledge of ML and AI models.
The project compares CNN and VGG16 by using both of them and creating plots and confusion matrices to compare their effectiveness.

## Installation
Use the package manager pip to install a couple of important libraries - opencv-python, numpy, matplotlib, scikit-learn, and tensorflow

```bash
pip install opencv
```
```bash
pip install numpy
```
```bash
pip install matplotlib
```
```bash
pip install scikit-learn
```
```bash
pip install tensorflow
```

Download the chest X-rays images dataset from the Kaggle website

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Project Outcomes
The project runs a comparison between the VGG16 and CNN models, and based on the output, the CNN model is more balanced and has more true positives for determining pneumonia than VGG16, which had more false negatives.

