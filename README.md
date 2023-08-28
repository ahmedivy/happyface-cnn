# 😀 Happy CNN

> Emotion Detection using Convolutional Neural Networks

This repository contains very simple CNN Architecture for emotion detection. The model is trained on a dataset of happy and unhappy faces. This is the same model that is taught in the first week of CNN course in [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) on Coursera. In course they uses Keras but I have implemented the same model using PyTorch.

## Table of Contents

- [😀 Happy CNN](#-happy-cnn)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Requirements](#requirements)
  - [Usage](#usage)
    - [Training](#training)
    - [Testing](#testing)
  - [Model Architecture](#model-architecture)
  - [Dataset](#dataset)
  - [Results](#results)
  - [References](#references)

## Introduction

The "Happy CNN" project aims to develop a deep learning model capable of recognizing happy faces in images. The model is implemented using PyTorch and follows a Convolutional Neural Network architecture.

## Requirements

- Python 3.x
- PyTorch
- h5py
- NumPy
- PIL (Python Imaging Library)

You can install the required dependencies using the following command:

```bash
pip install torch h5py numpy pillow
```

- `torch`: The PyTorch library for building and training neural networks.
- `h5py`: A package to interact with HDF5 (Hierarchical Data Format) files.
- `numpy`: A fundamental package for numerical operations in Python.
- `pillow`: The Python Imaging Library for opening, manipulating, and saving image files.

## Usage

### Training

To train the "Happy CNN" model, use the following command:

```bash
python model.py --train --epochs <number_of_epochs> --batch_size <batch_size>
```

### Testing

To test the "Happy CNN" model, use the following command:

```bash
python model.py --predict <path_to_image_file>
```

## Model Architecture

The "Happy CNN" model consists of several layers, including convolutional layers, batch normalization, ReLU activation, max-pooling, flattening, and fully connected layers. The architecture is designed to process images and make predictions about the emotion expressed in them.

## Dataset

The dataset used for training and testing the model is stored in HDF5 files (datasets/train_happy.h5 and datasets/test_happy.h5). It contains labeled images of happy and unhappy faces. The training and testing images are preprocessed and normalized before being fed into the model.

## Results

During training, the model's loss is monitored to track its convergence. After training, the model's weights are saved to model/happy-cnn.pth. The trained model can then be used for predicting whether a given image contains a happy face.

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [h5py Documentation](https://docs.h5py.org/en/stable/index.html)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [PIL Documentation](https://pillow.readthedocs.io/en/stable/index.html)

Feel free to contribute to the project by improving the model architecture, enhancing the dataset, or extending its functionality!
