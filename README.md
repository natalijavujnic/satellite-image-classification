# Satellite Image Classification Using CNN

A deep learning project focused on classifying satellite images using a Convolutional Neural Network (CNN) implemented with TensorFlow and Keras.

Developed as part of the **Neural Networks** course at the University of Belgrade, School of Electrical Engineering.

## Project Overview

The goal of this project is to develop and evaluate a CNN capable of classifying satellite images into four categories:

- **Cloudy** — Cloud-covered areas
- **Desert** — Desert landscapes
- **Green Area** — Vegetation-covered regions
- **Water** — Bodies of water

The dataset contains **5,631 satellite images**, divided into training, validation, and testing subsets.

## Dataset

The project uses the **Satellite Image Classification** dataset available on Kaggle.

The dataset consists of **5,631 satellite images** distributed across four classes:

| Class | Number of Images |
|---|---:|
| Cloudy | 1,500 |
| Desert | 1,131 |
| Green Area | 1,500 |
| Water | 1,500 |
| **Total** | **5,631** |

The images were divided into training, validation, and testing subsets, with data augmentation applied to improve model generalization.

**Dataset source:** [Satellite Image Classification — Kaggle](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification)

## Model Architecture

The CNN was built from scratch using the Keras Sequential API, without transfer learning or pretrained models.

The architecture includes:

- Data augmentation using random flipping, rotation, and zooming
- Input normalization
- Three convolutional layers with ReLU activation
- Three max-pooling layers
- Dropout regularization (20%)
- Fully connected layer with 128 neurons
- Softmax output layer for four-class classification

Early stopping was implemented to reduce overfitting.

## Results

The model achieved the following classification accuracy:

| Dataset | Accuracy |
|---|---:|
| Training | 99.38% |
| Testing | **98.55%** |

Model performance was evaluated using accuracy and loss curves, confusion matrices, and examples of correctly and incorrectly classified images.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Project Files

- `nmProj.py` — CNN implementation, training, evaluation, and visualizations.
- `NMProjekat.pdf` — Project report containing methodology, architecture, and experimental results.

## Authors

- Natalija Vujnić
- Anja Mihajlov

**University of Belgrade — School of Electrical Engineering**