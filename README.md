# CNN Image Classification – Cats vs Dogs

##  Overview
This project implements a **Convolutional Neural Network (CNN)** for **binary image classification** on the **Cats vs Dogs** dataset.  
The objective is to understand the **core components of CNNs** and the end-to-end workflow for training image classification models.

The project focuses on **foundational deep learning concepts** rather than transfer learning or large-scale vision models.

---

##  Key Concepts Covered
- Convolutional layers and feature extraction
- Pooling for spatial downsampling
- Non-linear activations (ReLU)
- Fully connected layers
- Binary classification using Sigmoid activation
- Data augmentation for improving generalization

---

## tech Stack
- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**

---

##  Project Structure
cnn-image-classification-cats-dogs/
├── cnn_model.ipynb
├── requirements.txt
└── README.md

---

##  Model Architecture
- Convolution + ReLU layers for feature extraction
- Max-Pooling layers for spatial reduction
- Fully connected layers for classification
- Output layer with **Sigmoid** activation

---

## Model Intuition
CNNs learn **spatial hierarchies of features** by applying convolutional filters over local regions of an image.

The model is trained using **Binary Cross-Entropy loss**:

\[
L = -[y \log(\hat{y}) + (1 - y)\log(1 - \hat{y})]
\]

This loss encourages the network to output accurate probabilities for the two classes.

---

##  How to Run
1. Install dependencies
   ```bash
   pip install -r requirements.txt

## Results
•	The CNN successfully learns visual patterns distinguishing cats and dogs
•	Data augmentation improves generalization
•	Demonstrates the effectiveness of CNNs for image classification tasks

## Notes
	•	This is a foundational computer vision project
	•	The CNN is trained from scratch
	•	No pretrained models or transfer learning are used
	•	Intended for learning and experimentation

